#!/usr/bin/env python
"""Run the frozen U5.R2AN0 paired positive-film recovery experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.positive_film import (  # noqa: E402
    PositiveFilmResponseOperator,
    finite_difference_jacobians,
    positive_film_operator_from_config,
)
from src.roll2film.positive_film_fitting import (  # noqa: E402
    PositiveFilmFitResult,
    fit_positive_film_response_operator,
)


CONFIG_SHA256 = "dd711c5e07344b60d5a954f0297bf513dc18f11a1c78e158eb7c65a7af4b89af"
REPORT_SCHEMA = "neuro-film.u5.r2an0.paired-positive-film-recovery-report.v1"
REPEAT_SCHEMA = "neuro-film.u5.r2an0.repeat-decision.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AN0 requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AN0 config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"] != "u5.r2an0-paired-positive-film-recovery-v1"
        or config["truth_operator_config_sha256"]
        != "3fa39f74faede3fac5767fca9f29a9ec61859d8b879a2ff95293163811a9fcf2"
        or config["claim_ceiling"].startswith("clean-room synthetic") is False
        or config["real_film_pixel_allowed"]
        or config["production_integration_allowed"]
        or config["stock_identity_claim_allowed"]
    ):
        raise ValueError("AN0 frozen contract mismatch")
    return config


def generate_paired_design(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate rows and their patch/illuminant/exposure group identities."""

    design = config["synthetic_design"]
    count = int(design["base_patch_count"])
    rng = np.random.default_rng(int(design["base_patch_seed"]))
    log_values = rng.uniform(
        float(design["base_patch_log2_minimum"]),
        float(design["base_patch_log2_maximum"]),
        size=(count, 3),
    )
    patches = np.exp2(log_values)
    # Deterministic neutral rows ensure the tone response is directly observed.
    neutral_count = min(12, count)
    neutral = np.geomspace(
        np.exp2(float(design["base_patch_log2_minimum"])),
        np.exp2(float(design["base_patch_log2_maximum"])),
        neutral_count,
    )
    patches[:neutral_count] = neutral[:, None]

    rows: list[np.ndarray] = []
    patch_indices: list[int] = []
    illuminant_indices: list[int] = []
    exposure_indices: list[int] = []
    for patch_index, patch in enumerate(patches):
        for illuminant_index, gains in enumerate(design["illuminant_gains"]):
            gain = np.asarray(gains, dtype=np.float64)
            for exposure_index, exposure_ev in enumerate(design["exposure_ev"]):
                rows.append(
                    np.clip(
                        patch * gain * np.exp2(float(exposure_ev)),
                        float(design["input_clip_minimum"]),
                        float(design["input_clip_maximum"]),
                    )
                )
                patch_indices.append(patch_index)
                illuminant_indices.append(illuminant_index)
                exposure_indices.append(exposure_index)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(patch_indices, dtype=np.int64),
        np.asarray(illuminant_indices, dtype=np.int64),
        np.asarray(exposure_indices, dtype=np.int64),
    )


def split_masks(
    config: dict[str, Any],
    patch_indices: np.ndarray,
    illuminant_indices: np.ndarray,
    exposure_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    split = config["group_split"]
    patch_development = np.isin(
        patch_indices % int(split["patch_modulus"]),
        np.asarray(split["development_patch_residues"]),
    )
    illuminant_development = np.isin(
        illuminant_indices,
        np.asarray(split["development_illuminant_indices"]),
    )
    exposure_development = np.isin(
        exposure_indices,
        np.asarray(split["development_exposure_indices"]),
    )
    development = patch_development & illuminant_development & exposure_development
    held_patch = ~patch_development
    held_illuminant = patch_development & ~illuminant_development
    held_exposure = (
        patch_development & illuminant_development & ~exposure_development
    )
    confirmation = held_patch | held_illuminant | held_exposure
    if np.any(development & confirmation) or not np.all(development | confirmation):
        raise RuntimeError("AN0 split is not an exact complement")
    return {
        "development": development,
        "confirmation": confirmation,
        "held_patch": held_patch,
        "held_illuminant": held_illuminant,
        "held_exposure": held_exposure,
    }


def _error_metrics(
    operator: PositiveFilmResponseOperator,
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    error = operator.apply(source) - target
    return {
        "rgb_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def _fit_record(result: PositiveFilmFitResult) -> dict[str, Any]:
    return {
        "model": result.model,
        "development_rgb_rmse": result.development_rgb_rmse,
        "development_maximum_absolute_error": (
            result.development_maximum_absolute_error
        ),
        "optimization_cost": result.optimization_cost,
        "optimality": result.optimality,
        "function_evaluations": result.function_evaluations,
        "restart_index": result.restart_index,
        "converged": result.converged,
        "operator": result.operator.to_dict(),
        "operator_sha256": _sha256(_canonical_json(result.operator.to_dict())),
    }


def _structure_metrics(
    operator: PositiveFilmResponseOperator, config: dict[str, Any]
) -> dict[str, float]:
    axis = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    grid = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    interior_axis = np.linspace(0.05, 0.95, 7, dtype=np.float64)
    interior = np.stack(
        np.meshgrid(interior_axis, interior_axis, interior_axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    output = operator.apply(grid)
    jacobians = finite_difference_jacobians(operator, interior, step=1e-6)
    return {
        "output_minimum": float(np.min(output)),
        "output_maximum": float(np.max(output)),
        "minimum_jacobian_determinant": float(
            np.min(np.linalg.det(jacobians))
        ),
        "capture_matrix_determinant": float(
            np.linalg.det(operator.capture_matrix)
        ),
        "scan_matrix_determinant": float(np.linalg.det(operator.scan_matrix)),
    }


def run_child(
    config: dict[str, Any],
    *,
    software_commit: str,
) -> dict[str, Any]:
    source, patch, illuminant, exposure = generate_paired_design(config)
    masks = split_masks(config, patch, illuminant, exposure)
    truth_raw = (
        ROOT / str(config["truth_operator_config"])
    ).read_bytes()
    if _sha256(truth_raw) != config["truth_operator_config_sha256"]:
        raise ValueError("AN0 truth config hash mismatch")
    truth_config = json.loads(truth_raw)
    source_before = source.copy()
    models = config["models"]
    fit_options = {
        "identity_mixture": float(models["matrix_identity_mixture"]),
        "restart_count": int(models["restart_count"]),
        "maximum_function_evaluations": int(
            models["maximum_function_evaluations"]
        ),
        "function_tolerance": float(models["function_tolerance"]),
        "parameter_tolerance": float(models["parameter_tolerance"]),
        "gradient_tolerance": float(models["gradient_tolerance"]),
        "seed": int(config["synthetic_design"]["base_patch_seed"]),
    }

    witnesses: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    gates = config["gates"]
    for witness_name in config["truth_witnesses"]:
        truth = positive_film_operator_from_config(
            truth_config["witnesses"][witness_name]
        )
        target = truth.apply(source)
        development_source = source[masks["development"]]
        development_target = target[masks["development"]]
        candidate = fit_positive_film_response_operator(
            development_source,
            development_target,
            model="two_matrix",
            **fit_options,
        )
        ablation = fit_positive_film_response_operator(
            development_source,
            development_target,
            model="one_matrix",
            **fit_options,
        )
        candidate_confirmation = _error_metrics(
            candidate.operator,
            source[masks["confirmation"]],
            target[masks["confirmation"]],
        )
        ablation_confirmation = _error_metrics(
            ablation.operator,
            source[masks["confirmation"]],
            target[masks["confirmation"]],
        )
        relative_gain = 1.0 - (
            candidate_confirmation["rgb_rmse"]
            / ablation_confirmation["rgb_rmse"]
        )
        partitions = {
            name: {
                "row_count": int(np.count_nonzero(masks[name])),
                "candidate": _error_metrics(
                    candidate.operator, source[masks[name]], target[masks[name]]
                ),
                "ablation": _error_metrics(
                    ablation.operator, source[masks[name]], target[masks[name]]
                ),
            }
            for name in ("held_patch", "held_illuminant", "held_exposure")
        }
        structure = _structure_metrics(candidate.operator, config)
        witness_checks = [
            {
                "name": f"{witness_name}.candidate_confirmation_rmse",
                "passed": candidate_confirmation["rgb_rmse"]
                <= float(gates["confirmation_rgb_rmse_maximum"]),
            },
            {
                "name": f"{witness_name}.gain_over_ablation",
                "passed": relative_gain
                >= float(gates["minimum_relative_gain_over_one_matrix_ablation"]),
            },
            {
                "name": f"{witness_name}.candidate_confirmation_max_error",
                "passed": candidate_confirmation["maximum_absolute_error"]
                <= float(gates["confirmation_maximum_absolute_error"]),
            },
            {
                "name": f"{witness_name}.output_cube",
                "passed": structure["output_minimum"]
                >= float(gates["output_minimum"]) - 1e-12
                and structure["output_maximum"]
                <= float(gates["output_maximum"]) + 1e-12,
            },
            {
                "name": f"{witness_name}.positive_jacobian",
                "passed": structure["minimum_jacobian_determinant"]
                > float(gates["minimum_finite_difference_jacobian_determinant"]),
            },
        ]
        checks.extend(witness_checks)
        witnesses[witness_name] = {
            "truth_operator_sha256": _sha256(_canonical_json(truth.to_dict())),
            "candidate_fit": _fit_record(candidate),
            "ablation_fit": _fit_record(ablation),
            "confirmation": {
                "row_count": int(np.count_nonzero(masks["confirmation"])),
                "candidate": candidate_confirmation,
                "ablation": ablation_confirmation,
                "relative_gain_over_ablation": relative_gain,
            },
            "confirmation_partitions": partitions,
            "candidate_structure": structure,
            "automatic_checks": witness_checks,
        }

    source_nonmutation = np.array_equal(source, source_before)
    checks.append({"name": "source_nonmutation", "passed": source_nonmutation})
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": CONFIG_SHA256,
        "truth_operator_config_sha256": config["truth_operator_config_sha256"],
        "design": {
            "row_count": int(source.shape[0]),
            "source_f64_sha256": _sha256(source.tobytes(order="C")),
            "patch_group_count": int(np.unique(patch).size),
            "illuminant_group_count": int(np.unique(illuminant).size),
            "exposure_group_count": int(np.unique(exposure).size),
            "development_row_count": int(np.count_nonzero(masks["development"])),
            "confirmation_row_count": int(np.count_nonzero(masks["confirmation"])),
            "held_patch_row_count": int(np.count_nonzero(masks["held_patch"])),
            "held_illuminant_row_count": int(
                np.count_nonzero(masks["held_illuminant"])
            ),
            "held_exposure_row_count": int(
                np.count_nonzero(masks["held_exposure"])
            ),
        },
        "witnesses": witnesses,
        "automatic_checks": checks,
        "automatic_pass": all(bool(check["passed"]) for check in checks),
        "claim_ceiling": config["claim_ceiling"],
    }
    return report


def _child_command(
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--expected-config-sha256",
        CONFIG_SHA256,
        "--child-output",
        str(output_path),
        "--software-commit",
        software_commit,
    ]


def run_parent(config_path: Path, output_dir: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    config = load_config(config_path, expected_sha256=CONFIG_SHA256)
    software_commit = _git_commit()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths = [output_dir / "report_a.json", output_dir / "report_b.json"]
    for path in report_paths:
        subprocess.run(
            _child_command(config_path, path, software_commit),
            cwd=ROOT,
            check=True,
        )
    report_bytes = [path.read_bytes() for path in report_paths]
    reports = [json.loads(value) for value in report_bytes]
    if any(report["software_commit"] != software_commit for report in reports):
        raise RuntimeError("AN0 child software commit mismatch")
    byte_identical = report_bytes[0] == report_bytes[1]
    automatic_pass = byte_identical and all(
        bool(report["automatic_pass"]) for report in reports
    )
    decision = {
        "schema": REPEAT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": CONFIG_SHA256,
        "report_sha256": [_sha256(value) for value in report_bytes],
        "reports_byte_identical": byte_identical,
        "automatic_pass": automatic_pass,
        "decision": (
            "advance_paired_positive_film_recovery"
            if automatic_pass
            else "close_or_diagnose_paired_positive_film_recovery"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    decision_bytes = _canonical_json(decision)
    _atomic_write(output_dir / "repeat_decision.json", decision_bytes)
    print(
        json.dumps(
            {
                "automatic_pass": automatic_pass,
                "decision": decision["decision"],
                "report_sha256": decision["report_sha256"],
                "repeat_decision_sha256": _sha256(decision_bytes),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2an0_paired_positive_film_recovery_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256",
        default=CONFIG_SHA256,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--software-commit")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(
        args.config, expected_sha256=args.expected_config_sha256
    )
    if args.child_output is not None:
        if not args.software_commit:
            raise ValueError("AN0 child requires --software-commit")
        report = run_child(config, software_commit=args.software_commit)
        _atomic_write(args.child_output, _canonical_json(report))
        return 0
    if args.output_dir is None:
        raise ValueError("AN0 parent requires --output-dir")
    run_parent(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
