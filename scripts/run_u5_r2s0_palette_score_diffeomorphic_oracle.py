#!/usr/bin/env python
"""Run the frozen U5.R2S0 palette-score diffeomorphic oracle."""

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

from src.roll2film.cube_diffeomorphic_flow import (  # noqa: E402
    CubeDiffeomorphicColourFlow,
    finite_difference_jacobians,
)
from src.roll2film.palette_score_flow import (  # noqa: E402
    DiagonalGaussianMixturePalette,
    palette_score_operator,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _grid(axis_size: int) -> np.ndarray:
    axis = (np.arange(axis_size, dtype=np.float64) + 0.5) / axis_size
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _best_affine_residual(source: np.ndarray, target: np.ndarray) -> float:
    design = np.column_stack((np.ones(len(source)), source))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    prediction = design @ coefficients
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def run_audit(
    config: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    candidate = config["candidate"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    confirmation = _grid(int(evaluation["confirmation_grid_axis_size"]))
    attraction = _grid(int(evaluation["palette_attraction_grid_axis_size"]))
    jacobian_axis = np.linspace(
        float(evaluation["jacobian_domain_minimum"]),
        float(evaluation["jacobian_domain_maximum"]),
        int(evaluation["jacobian_grid_axis_size"]),
        dtype=np.float64,
    )
    jacobian_points = np.stack(
        np.meshgrid(jacobian_axis, jacobian_axis, jacobian_axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    identity = CubeDiffeomorphicColourFlow.identity(
        axis_size=int(candidate["velocity_grid_axis_size"]),
        integration_steps=int(candidate["integration_steps"]),
    )
    identity_output = identity.apply(confirmation)
    identity_error = float(np.max(np.abs(identity_output - confirmation)))

    palette_reports: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    for name, payload in config["palettes"].items():
        palette = DiagonalGaussianMixturePalette.from_config(payload)
        operator = palette_score_operator(
            palette,
            axis_size=int(candidate["velocity_grid_axis_size"]),
            integration_steps=int(candidate["integration_steps"]),
            coefficient_vector_norm_cap=float(
                candidate["coefficient_vector_norm_cap"]
            ),
        )
        output = operator.apply(confirmation)
        outputs[name] = output
        jacobians = finite_difference_jacobians(
            operator,
            jacobian_points,
            step=float(evaluation["finite_difference_step"]),
        )
        determinants = np.linalg.det(jacobians)
        spectral_norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
        inverse_error = float(
            np.max(np.abs(operator.inverse(output) - confirmation))
        )
        replay = CubeDiffeomorphicColourFlow.from_dict(
            json.loads(json.dumps(operator.to_dict(), sort_keys=True))
        )
        replay_error = float(np.max(np.abs(replay.apply(confirmation) - output)))
        split = int(evaluation["partition_split_index"])
        partition = np.concatenate(
            (
                operator.apply(confirmation[:split]),
                operator.apply(confirmation[split:]),
            )
        )
        partition_error = float(np.max(np.abs(partition - output)))
        attraction_output = operator.apply(attraction)
        log_density_gain = float(
            np.mean(
                palette.log_density(attraction_output)
                - palette.log_density(attraction)
            )
        )
        coefficient_norms = np.linalg.norm(operator.velocity_grid, axis=-1)
        metrics = {
            "minimum_output": float(np.min(output)),
            "maximum_output": float(np.max(output)),
            "identity_rmse": _rmse(output, confirmation),
            "best_affine_residual_rmse": _best_affine_residual(
                confirmation, output
            ),
            "mean_log_density_gain": log_density_gain,
            "minimum_jacobian_determinant": float(np.min(determinants)),
            "p01_jacobian_determinant": float(
                np.quantile(determinants, 0.01)
            ),
            "maximum_jacobian_spectral_norm": float(
                np.max(spectral_norms)
            ),
            "maximum_inverse_error": inverse_error,
            "maximum_replay_error": replay_error,
            "maximum_partition_error": partition_error,
            "maximum_coefficient_vector_norm": float(
                np.max(coefficient_norms)
            ),
        }
        gate_results = {
            "range": (
                metrics["minimum_output"] >= float(gates["minimum_output"])
                and metrics["maximum_output"] <= float(gates["maximum_output"])
            ),
            "positive_jacobian": metrics["minimum_jacobian_determinant"]
            > float(gates["minimum_jacobian_determinant_exclusive"]),
            "bounded_jacobian_norm": metrics[
                "maximum_jacobian_spectral_norm"
            ]
            <= float(gates["maximum_jacobian_spectral_norm"]),
            "inverse": metrics["maximum_inverse_error"]
            <= float(gates["maximum_inverse_error"]),
            "replay": metrics["maximum_replay_error"]
            <= float(gates["maximum_replay_error"]),
            "partition": metrics["maximum_partition_error"]
            <= float(gates["maximum_partition_error"]),
            "style_strength": metrics["identity_rmse"]
            >= float(gates["minimum_identity_rmse_each_palette"]),
            "non_affine": metrics["best_affine_residual_rmse"]
            >= float(gates["minimum_best_affine_residual_rmse_each_palette"]),
            "palette_attraction": metrics["mean_log_density_gain"]
            >= float(gates["minimum_mean_log_density_gain_each_palette"]),
            "coefficient_bound": metrics["maximum_coefficient_vector_norm"]
            <= float(gates["maximum_coefficient_vector_norm"]),
        }
        gate_results["all"] = bool(all(gate_results.values()))
        palette_reports[name] = {
            "metrics": metrics,
            "gate_results": gate_results,
            "operator": operator.to_dict(),
        }

    pairwise = {}
    names = list(outputs)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = _rmse(
                outputs[left], outputs[right]
            )
    minimum_pairwise = float(min(pairwise.values()))
    identity_pass = identity_error <= float(
        gates["identity_maximum_absolute_error"]
    )
    palette_pass = all(
        bool(report["gate_results"]["all"])
        for report in palette_reports.values()
    )
    reference_pass = minimum_pairwise >= float(
        gates["minimum_pairwise_operator_output_rmse"]
    )
    all_pass = bool(identity_pass and palette_pass and reference_pass)
    if all_pass:
        branch = "all_gates_pass"
    elif not palette_pass:
        failed = {
            gate
            for report in palette_reports.values()
            for gate, passed in report["gate_results"].items()
            if gate != "all" and not passed
        }
        if failed & {
            "range",
            "positive_jacobian",
            "bounded_jacobian_norm",
            "inverse",
            "replay",
            "partition",
            "coefficient_bound",
        }:
            branch = "structure_gate_fails"
        elif "palette_attraction" in failed:
            branch = "attraction_gate_fails"
        else:
            branch = "style_or_reference_gate_fails"
    else:
        branch = "style_or_reference_gate_fails"
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "identity_maximum_absolute_error": identity_error,
        "palettes": palette_reports,
        "pairwise_output_rmse": pairwise,
        "minimum_pairwise_output_rmse": minimum_pairwise,
        "gates": gates,
        "gate_results": {
            "identity": identity_pass,
            "all_palettes": palette_pass,
            "reference_separation": reference_pass,
            "all": all_pass,
        },
        "decision_branch": branch,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2s0_palette_score_diffeomorphic_oracle_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--software-commit",
        default=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    report = run_audit(
        json.loads(config_bytes),
        config_sha256=_sha256(config_bytes),
        software_commit=str(args.software_commit),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    args.output.write_bytes(payload)
    print(json.dumps(report["gate_results"], sort_keys=True))
    print(f"decision_branch={report['decision_branch']}")
    print(f"report_sha256={_sha256(payload)}")


if __name__ == "__main__":
    main()
