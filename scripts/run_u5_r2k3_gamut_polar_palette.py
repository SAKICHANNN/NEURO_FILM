#!/usr/bin/env python
"""Run the frozen U5.R2K3 gamut-polar palette representation audit."""

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

from src.roll2film.gamut_polar_palette import (  # noqa: E402
    GamutPolarPaletteOperator,
    finite_difference_jacobians,
    operator_from_config,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _grid(axis_size: int, *, interior: bool) -> np.ndarray:
    if interior:
        axis = np.linspace(0.05, 0.95, axis_size, dtype=np.float64)
    else:
        axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _best_affine_prediction(
    source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    design = np.column_stack((np.ones(len(source)), source))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return design @ coefficients


def run_audit(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    candidate = config["candidate"]
    gates = config["gates"]
    audit_grid = _grid(int(config["audit_grid_axis_size"]), interior=False)
    jacobian_grid = _grid(
        int(config["jacobian_grid_axis_size"]), interior=True
    )
    neutral_values = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    neutral = np.repeat(neutral_values[:, None], 3, axis=1)
    endpoints = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64
    )
    reports: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    for name, witness in config["witnesses"].items():
        operator = operator_from_config(candidate, witness)
        output = operator.apply(audit_grid)
        outputs[name] = output
        restored = operator.inverse(output)
        neutral_output = operator.apply(neutral)
        endpoint_output = operator.apply(endpoints)
        jacobians = finite_difference_jacobians(
            operator,
            jacobian_grid,
            step=float(config["finite_difference_step"]),
        )
        determinants = np.linalg.det(jacobians)
        spectral_norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
        replay = GamutPolarPaletteOperator.from_dict(
            json.loads(json.dumps(operator.to_dict(), sort_keys=True))
        )
        split = len(audit_grid) // 3
        partitioned = np.concatenate(
            (
                operator.apply(audit_grid[:split]),
                operator.apply(audit_grid[split : 2 * split]),
                operator.apply(audit_grid[2 * split :]),
            )
        )
        affine = _best_affine_prediction(audit_grid, output)
        reports[name] = {
            "output_minimum": float(np.min(output)),
            "output_maximum": float(np.max(output)),
            "identity_rgb_rmse": _rmse(output, audit_grid),
            "residual_after_best_affine_rgb_rmse": _rmse(output, affine),
            "neutral_axis_maximum_chroma": float(
                np.max(np.ptp(neutral_output, axis=1))
            ),
            "endpoint_maximum_absolute_error": float(
                np.max(np.abs(endpoint_output - endpoints))
            ),
            "inverse_roundtrip_maximum_absolute_error": float(
                np.max(np.abs(restored - audit_grid))
            ),
            "minimum_finite_difference_jacobian_determinant": float(
                np.min(determinants)
            ),
            "maximum_finite_difference_jacobian_spectral_norm": float(
                np.max(spectral_norms)
            ),
            "serialization_replay_maximum_absolute_error": float(
                np.max(np.abs(replay.apply(audit_grid) - output))
            ),
            "partition_maximum_absolute_error": float(
                np.max(np.abs(partitioned - output))
            ),
        }
    pairwise = []
    names = list(outputs)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "rgb_rmse": _rmse(outputs[left], outputs[right]),
                }
            )
    checks = {
        "identity_exact": (
            reports["identity"]["identity_rgb_rmse"]
            <= gates["identity_maximum_absolute_error"]
        ),
        "neutral_axis_exact": all(
            report["neutral_axis_maximum_chroma"]
            <= gates["neutral_axis_maximum_chroma"]
            for report in reports.values()
        ),
        "endpoints_exact": all(
            report["endpoint_maximum_absolute_error"]
            <= gates["endpoint_maximum_absolute_error"]
            for report in reports.values()
        ),
        "bounded": all(
            report["output_minimum"] >= gates["output_minimum"]
            and report["output_maximum"] <= gates["output_maximum"]
            for report in reports.values()
        ),
        "inverse_roundtrip": all(
            report["inverse_roundtrip_maximum_absolute_error"]
            <= gates["inverse_roundtrip_maximum_absolute_error"]
            for report in reports.values()
        ),
        "positive_jacobian": all(
            report["minimum_finite_difference_jacobian_determinant"]
            > gates["minimum_finite_difference_jacobian_determinant"]
            for report in reports.values()
        ),
        "bounded_jacobian_norm": all(
            report["maximum_finite_difference_jacobian_spectral_norm"]
            <= gates["maximum_finite_difference_jacobian_spectral_norm"]
            for report in reports.values()
        ),
        "pairwise_diversity": min(
            record["rgb_rmse"] for record in pairwise
        )
        >= gates["minimum_pairwise_witness_rgb_rmse"],
        "identity_distance": all(
            report["identity_rgb_rmse"]
            >= gates["minimum_non_identity_rgb_rmse"]
            for name, report in reports.items()
            if name != "identity"
        ),
        "non_affine": all(
            report["residual_after_best_affine_rgb_rmse"]
            >= gates["minimum_residual_after_best_affine_rgb_fit"]
            for name, report in reports.items()
            if name != "identity"
        ),
        "serialization_replay": all(
            report["serialization_replay_maximum_absolute_error"]
            <= gates["serialization_replay_maximum_absolute_error"]
            for report in reports.values()
        ),
        "partition_parity": all(
            report["partition_maximum_absolute_error"]
            <= gates["partition_maximum_absolute_error"]
            for report in reports.values()
        ),
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": config_sha256,
        "audit_grid_points": len(audit_grid),
        "jacobian_grid_points": len(jacobian_grid),
        "witnesses": reports,
        "pairwise_witness_rgb_rmse": pairwise,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2k3_gamut_polar_palette_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2k3_gamut_polar_palette_v1/report.json",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    report = run_audit(json.loads(config_bytes), _sha256(config_bytes))
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(encoded),
                "all_checks_passed": report["all_checks_passed"],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
