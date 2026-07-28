#!/usr/bin/env python
"""Run the frozen U5.R2AK1 non-autonomous colour-flow capacity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.cube_diffeomorphic_flow import (  # noqa: E402
    CubeDiffeomorphicColourFlow,
    _smoothness_loss,
    finite_difference_jacobians,
)
from src.roll2film.time_dependent_cube_flow import (  # noqa: E402
    TimeDependentCubeColourFlow,
    _sample_grid_torch_vectorized,
    fit_time_dependent_cube_colour_flow,
)


REPORT_SCHEMA = "neuro-film.u5.r2ak1.capacity-report.v1"
REPEAT_SCHEMA = "neuro-film.u5.r2ak1.repeat-decision.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise RuntimeError("formal AK1 audit requires a clean tracked worktree")


def _midpoint_grid(axis_size: int) -> np.ndarray:
    axis = (np.arange(axis_size, dtype=np.float64) + 0.5) / axis_size
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _corners() -> np.ndarray:
    return np.stack(
        np.meshgrid(
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)


def _source_fields(
    config: dict[str, Any],
) -> tuple[CubeDiffeomorphicColourFlow, CubeDiffeomorphicColourFlow]:
    truth = config["synthetic_truth"]
    axis_size = int(truth["source_grid_axis_size"])
    axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    cube = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    )
    red, green, blue = np.moveaxis(cube, -1, 0)
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    scale = float(truth["source_field_scale"])

    field_a = np.empty_like(cube)
    field_a[..., 0] = scale * (
        1.20 * (0.52 - luma) + 0.25 * (green - blue)
    )
    field_a[..., 1] = scale * (
        0.55 * (0.50 - luma) - 0.18 * (red - blue)
    )
    field_a[..., 2] = scale * (
        0.18 * (0.48 - luma) - 0.12 * (red - green)
    )

    field_b = np.empty_like(cube)
    field_b[..., 0] = scale * (
        0.70 * (green - blue) + 0.15 * (0.50 - luma)
    )
    field_b[..., 1] = scale * (0.55 * (blue - red))
    field_b[..., 2] = scale * (
        0.80 * (red - green) - 0.20 * (0.50 - luma)
    )
    integration_steps = int(truth["source_integration_steps"])
    return (
        CubeDiffeomorphicColourFlow(
            field_a, integration_steps=integration_steps
        ),
        CubeDiffeomorphicColourFlow(
            field_b, integration_steps=integration_steps
        ),
    )


def _target(
    name: str,
    rgb: np.ndarray,
    field_a: CubeDiffeomorphicColourFlow,
    field_b: CubeDiffeomorphicColourFlow,
) -> np.ndarray:
    if name == "field_a_then_field_b":
        return field_b.apply(field_a.apply(rgb))
    if name == "field_b_then_field_a":
        return field_a.apply(field_b.apply(rgb))
    raise ValueError(f"unsupported frozen target: {name}")


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _fit_candidate(
    source: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
) -> TimeDependentCubeColourFlow:
    candidate = config["candidate"]
    fit = config["fit"]
    return fit_time_dependent_cube_colour_flow(
        source,
        target,
        axis_size=int(candidate["velocity_grid_axis_size"]),
        integration_steps=int(candidate["integration_steps"]),
        maximum_absolute_coefficient=float(
            candidate["maximum_absolute_coefficient"]
        ),
        seed=int(fit["seed"]),
        steps=int(fit["steps"]),
        learning_rate=float(fit["learning_rate"]),
        coefficient_l2=float(fit["coefficient_l2"]),
        spatial_smoothness_l2=float(fit["spatial_smoothness_l2"]),
        temporal_smoothness_l2=float(fit["temporal_smoothness_l2"]),
        gradient_clip_norm=float(fit["gradient_clip_norm"]),
        thread_count=int(fit["thread_count"]),
    )


def _fit_stationary(
    source: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
    *,
    control_name: str,
) -> CubeDiffeomorphicColourFlow:
    control = config["stationary_controls"][control_name]
    fit = config["fit"]
    axis_size = int(control["velocity_grid_axis_size"])
    integration_steps = int(control["integration_steps"])
    coefficient_cap = float(
        config["candidate"]["maximum_absolute_coefficient"]
    )
    torch.manual_seed(int(fit["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(fit["thread_count"]))
    source_tensor = torch.from_numpy(source.reshape(-1, 3).copy())
    target_tensor = torch.from_numpy(target.reshape(-1, 3).copy())
    grid = torch.zeros(
        (axis_size, axis_size, axis_size, 3),
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        [grid], lr=float(fit["learning_rate"])
    )
    step = 1.0 / integration_steps

    def velocity(values: torch.Tensor) -> torch.Tensor:
        return (
            values
            * (1.0 - values)
            * _sample_grid_torch_vectorized(values, grid)
        )

    for _ in range(int(fit["steps"])):
        optimizer.zero_grad(set_to_none=True)
        current = source_tensor
        for _integration_index in range(integration_steps):
            k1 = velocity(current)
            k2 = velocity(current + 0.5 * step * k1)
            k3 = velocity(current + 0.5 * step * k2)
            k4 = velocity(current + step * k3)
            current = current + (step / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
        loss = torch.mean((current - target_tensor) ** 2)
        coefficient_l2 = float(fit["coefficient_l2"])
        if coefficient_l2:
            loss = loss + coefficient_l2 * torch.mean(grid**2)
        smoothness_l2 = float(fit["spatial_smoothness_l2"])
        if smoothness_l2:
            loss = loss + smoothness_l2 * _smoothness_loss(grid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [grid], float(fit["gradient_clip_norm"])
        )
        optimizer.step()
        with torch.no_grad():
            grid.clamp_(-coefficient_cap, coefficient_cap)
    return CubeDiffeomorphicColourFlow(
        velocity_grid=grid.detach().cpu().numpy(),
        integration_steps=integration_steps,
    )


def _operator_metrics(
    operator: Any,
    *,
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    confirm_source: np.ndarray,
    confirm_target: np.ndarray,
    jacobian_points: np.ndarray,
    finite_difference_step: float,
) -> dict[str, Any]:
    prediction = operator.apply(confirm_source)
    fit_prediction = operator.apply(fit_source)
    jacobians = finite_difference_jacobians(
        operator,
        jacobian_points,
        step=finite_difference_step,
    )
    determinants = np.linalg.det(jacobians)
    spectral_norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
    replay = type(operator).from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    split = len(confirm_source) // 3
    partitioned = np.concatenate(
        (
            operator.apply(confirm_source[:split]),
            operator.apply(confirm_source[split : 2 * split]),
            operator.apply(confirm_source[2 * split :]),
        )
    )
    if isinstance(operator, TimeDependentCubeColourFlow):
        coefficient = float(np.max(np.abs(operator.control_grids)))
        axis_size = int(operator.control_grids.shape[1])
        parameter_count = int(operator.control_grids.size)
    else:
        coefficient = float(np.max(np.abs(operator.velocity_grid)))
        axis_size = int(operator.velocity_grid.shape[0])
        parameter_count = int(operator.velocity_grid.size)
    return {
        "velocity_grid_axis_size": axis_size,
        "parameter_count": parameter_count,
        "integration_steps": int(operator.integration_steps),
        "maximum_absolute_coefficient": coefficient,
        "fit_rgb_rmse": _rmse(fit_prediction, fit_target),
        "confirmation_rgb_rmse": _rmse(prediction, confirm_target),
        "confirmation_maximum_absolute_error": float(
            np.max(np.abs(prediction - confirm_target))
        ),
        "endpoint_maximum_absolute_error": float(
            np.max(np.abs(operator.apply(_corners()) - _corners()))
        ),
        "output_minimum": float(np.min(prediction)),
        "output_maximum": float(np.max(prediction)),
        "minimum_finite_difference_jacobian_determinant": float(
            np.min(determinants)
        ),
        "maximum_finite_difference_jacobian_spectral_norm": float(
            np.max(spectral_norms)
        ),
        "inverse_roundtrip_maximum_absolute_error": float(
            np.max(np.abs(operator.inverse(prediction) - confirm_source))
        ),
        "serialization_replay_maximum_absolute_error": float(
            np.max(np.abs(replay.apply(confirm_source) - prediction))
        ),
        "partition_maximum_absolute_error": float(
            np.max(np.abs(partitioned - prediction))
        ),
    }


def _reconstruct_checks(report: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["gates"]
    target_rows = report["targets"]
    candidate_rows = [row["candidate"] for row in target_rows.values()]
    return {
        "target_noncommutation": (
            report["target_pair_rgb_rmse"]
            >= float(
                config["synthetic_truth"][
                    "minimum_required_target_pair_rmse"
                ]
            )
        ),
        "identity_exact": (
            report["identity_confirmation_maximum_absolute_error"]
            <= float(
                gates["identity_confirmation_maximum_absolute_error"]
            )
        ),
        "candidate_fidelity": all(
            row["confirmation_rgb_rmse"]
            <= float(gates["maximum_candidate_confirmation_rgb_rmse"])
            for row in candidate_rows
        ),
        "stationary_k4_improvement": all(
            row["candidate_improvement_over_stationary_k4_fraction"]
            >= float(
                gates[
                    "minimum_candidate_improvement_over_stationary_k4_fraction"
                ]
            )
            for row in target_rows.values()
        ),
        "stationary_k5_efficiency": all(
            row["candidate_to_stationary_k5_rmse_ratio"]
            <= float(
                gates["maximum_candidate_to_stationary_k5_rmse_ratio"]
            )
            for row in target_rows.values()
        ),
        "endpoints_fixed": all(
            row["endpoint_maximum_absolute_error"]
            <= float(gates["endpoint_maximum_absolute_error"])
            for row in candidate_rows
        ),
        "bounded": all(
            row["output_minimum"] >= float(gates["output_minimum"])
            and row["output_maximum"] <= float(gates["output_maximum"])
            for row in candidate_rows
        ),
        "coefficient_bound": all(
            row["maximum_absolute_coefficient"]
            <= float(gates["maximum_absolute_coefficient"])
            for row in candidate_rows
        ),
        "positive_jacobian": all(
            row["minimum_finite_difference_jacobian_determinant"]
            >= float(
                gates["minimum_finite_difference_jacobian_determinant"]
            )
            for row in candidate_rows
        ),
        "bounded_jacobian_norm": all(
            row["maximum_finite_difference_jacobian_spectral_norm"]
            <= float(
                gates["maximum_finite_difference_jacobian_spectral_norm"]
            )
            for row in candidate_rows
        ),
        "inverse_roundtrip": all(
            row["inverse_roundtrip_maximum_absolute_error"]
            <= float(
                gates["inverse_roundtrip_maximum_absolute_error"]
            )
            for row in candidate_rows
        ),
        "serialization_replay": all(
            row["serialization_replay_maximum_absolute_error"]
            <= float(
                gates["serialization_replay_maximum_absolute_error"]
            )
            for row in candidate_rows
        ),
        "partition_parity": all(
            row["partition_maximum_absolute_error"]
            <= float(gates["partition_maximum_absolute_error"])
            for row in candidate_rows
        ),
    }


def run_child_audit(
    config: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    fit_source = _midpoint_grid(int(config["fit"]["grid_axis_size"]))
    confirm_source = _midpoint_grid(
        int(config["confirmation_grid_axis_size"])
    )
    jacobian_axis = np.linspace(
        0.05,
        0.95,
        int(config["jacobian_grid_axis_size"]),
        dtype=np.float64,
    )
    jacobian_points = np.stack(
        np.meshgrid(
            jacobian_axis,
            jacobian_axis,
            jacobian_axis,
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    field_a, field_b = _source_fields(config)
    target_names = list(config["synthetic_truth"]["targets"])
    confirm_targets = {
        name: _target(name, confirm_source, field_a, field_b)
        for name in target_names
    }
    target_pair_rmse = _rmse(
        confirm_targets[target_names[0]],
        confirm_targets[target_names[1]],
    )
    minimum_noncommutation = float(
        config["synthetic_truth"]["minimum_required_target_pair_rmse"]
    )
    if target_pair_rmse < minimum_noncommutation:
        raise RuntimeError(
            "frozen truth does not meet its noncommutation pre-fit gate"
        )

    target_reports: dict[str, Any] = {}
    for name in target_names:
        fit_target = _target(name, fit_source, field_a, field_b)
        confirm_target = confirm_targets[name]
        candidate = _fit_candidate(fit_source, fit_target, config)
        stationary_k4 = _fit_stationary(
            fit_source,
            fit_target,
            config,
            control_name="lower_parameter_control",
        )
        stationary_k5 = _fit_stationary(
            fit_source,
            fit_target,
            config,
            control_name="upper_parameter_control",
        )
        common = {
            "fit_source": fit_source,
            "fit_target": fit_target,
            "confirm_source": confirm_source,
            "confirm_target": confirm_target,
            "jacobian_points": jacobian_points,
            "finite_difference_step": float(
                config["finite_difference_step"]
            ),
        }
        candidate_metrics = _operator_metrics(candidate, **common)
        k4_metrics = _operator_metrics(stationary_k4, **common)
        k5_metrics = _operator_metrics(stationary_k5, **common)
        k4_rmse = float(k4_metrics["confirmation_rgb_rmse"])
        k5_rmse = float(k5_metrics["confirmation_rgb_rmse"])
        candidate_rmse = float(candidate_metrics["confirmation_rgb_rmse"])
        target_reports[name] = {
            "candidate": candidate_metrics,
            "stationary_k4": k4_metrics,
            "stationary_k5": k5_metrics,
            "candidate_improvement_over_stationary_k4_fraction": (
                (k4_rmse - candidate_rmse) / k4_rmse
            ),
            "candidate_to_stationary_k5_rmse_ratio": (
                candidate_rmse / k5_rmse
            ),
        }

    identity = TimeDependentCubeColourFlow.identity(
        axis_size=int(config["candidate"]["velocity_grid_axis_size"]),
        integration_steps=int(config["candidate"]["integration_steps"]),
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "fit_grid_points": int(len(fit_source)),
        "confirmation_grid_points": int(len(confirm_source)),
        "jacobian_grid_points": int(len(jacobian_points)),
        "target_pair_rgb_rmse": target_pair_rmse,
        "identity_confirmation_maximum_absolute_error": float(
            np.max(
                np.abs(identity.apply(confirm_source) - confirm_source)
            )
        ),
        "targets": target_reports,
        "checks": {},
        "all_checks_passed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["checks"] = _reconstruct_checks(report, config)
    report["all_checks_passed"] = all(report["checks"].values())
    return report


def validate_report(
    report: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha256: str,
    software_commit: str,
) -> None:
    expected_top = {
        "schema",
        "schema_version",
        "experiment_id",
        "software_commit",
        "config_sha256",
        "fit_grid_points",
        "confirmation_grid_points",
        "jacobian_grid_points",
        "target_pair_rgb_rmse",
        "identity_confirmation_maximum_absolute_error",
        "targets",
        "checks",
        "all_checks_passed",
        "claim_ceiling",
    }
    if set(report) != expected_top:
        raise ValueError("AK1 report has unexpected top-level fields")
    if (
        report["schema"] != REPORT_SCHEMA
        or report["schema_version"] != 1
        or report["experiment_id"] != config["experiment_id"]
        or report["software_commit"] != software_commit
        or report["config_sha256"] != config_sha256
        or report["claim_ceiling"] != config["claim_ceiling"]
        or list(report["targets"]) != list(
            config["synthetic_truth"]["targets"]
        )
    ):
        raise ValueError("AK1 report identity mismatch")
    reconstructed = _reconstruct_checks(report, config)
    if report["checks"] != reconstructed:
        raise ValueError("AK1 report checks do not reconstruct")
    if report["all_checks_passed"] is not all(reconstructed.values()):
        raise ValueError("AK1 report aggregate decision does not reconstruct")


def _child_main(args: argparse.Namespace) -> int:
    _require_clean_tracked_worktree()
    before = args.config.read_bytes()
    config_sha256 = _sha256(before)
    if config_sha256 != args.expected_config_sha256:
        raise RuntimeError("AK1 config hash mismatch before audit")
    software_commit = _git_commit()
    if software_commit != args.expected_software_commit:
        raise RuntimeError("AK1 software commit mismatch")
    config = json.loads(before)
    report = run_child_audit(
        config,
        config_sha256=config_sha256,
        software_commit=software_commit,
    )
    validate_report(
        report,
        config=config,
        config_sha256=config_sha256,
        software_commit=software_commit,
    )
    after = args.config.read_bytes()
    if before != after:
        raise RuntimeError("AK1 config bytes changed during audit")
    _atomic_write(args.child_output, _canonical_json(report))
    return 0


def _parent_main(args: argparse.Namespace) -> int:
    _require_clean_tracked_worktree()
    config_bytes = args.config.read_bytes()
    config_sha256 = _sha256(config_bytes)
    if (
        args.expected_config_sha256
        and config_sha256 != args.expected_config_sha256
    ):
        raise RuntimeError("AK1 config does not match expected frozen hash")
    config = json.loads(config_bytes)
    software_commit = _git_commit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_paths = [
        args.output_dir / "run_a" / "report.json",
        args.output_dir / "run_b" / "report.json",
    ]
    for report_path in report_paths:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config.resolve()),
            "--expected-config-sha256",
            config_sha256,
            "--expected-software-commit",
            software_commit,
            "--child-output",
            str(report_path.resolve()),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    first = report_paths[0].read_bytes()
    second = report_paths[1].read_bytes()
    if first != second:
        raise RuntimeError("AK1 independent reports are not byte-identical")
    report = json.loads(first)
    validate_report(
        report,
        config=config,
        config_sha256=config_sha256,
        software_commit=software_commit,
    )
    if args.config.read_bytes() != config_bytes:
        raise RuntimeError("AK1 config bytes changed across repeat audit")
    decision = {
        "schema": REPEAT_SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "report_sha256": _sha256(first),
        "independent_report_count": 2,
        "reports_byte_identical": True,
        "automatic_checks": report["checks"],
        "automatic_pass": report["all_checks_passed"],
        "decision": (
            "retain_time_dependent_representation"
            if report["all_checks_passed"]
            else "close_fixed_time_dependent_representation"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    decision_bytes = _canonical_json(decision)
    _atomic_write(args.output_dir / "repeat_decision.json", decision_bytes)
    print(
        json.dumps(
            {
                "report_sha256": _sha256(first),
                "repeat_decision_sha256": _sha256(decision_bytes),
                "automatic_pass": report["all_checks_passed"],
                "decision": decision["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2ak1_time_dependent_cube_flow_capacity_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2ak1_time_dependent_cube_flow_capacity_v1"
        ),
    )
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-software-commit")
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    if args.child_output is not None:
        if not args.expected_config_sha256 or not args.expected_software_commit:
            parser.error(
                "child mode requires expected config hash and software commit"
            )
        return _child_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
