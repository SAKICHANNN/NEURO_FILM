#!/usr/bin/env python
"""Run the frozen U5.R2K1 triangular monotone coupling audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.density_domain import operator_from_config  # noqa: E402
from src.roll2film.monotone_coupling import (  # noqa: E402
    TriangularMonotoneCouplingOperator,
    finite_difference_jacobians,
    fit_triangular_monotone_coupling,
)
from src.roll2film.positive_film import (  # noqa: E402
    positive_film_operator_from_config,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _grid(axis_size: int) -> np.ndarray:
    axis = (np.arange(axis_size, dtype=np.float64) + 0.5) / axis_size
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )


def _target_functions(
    config: dict[str, Any],
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    density = json.loads(
        (ROOT / "configs/u5_r2e0_density_domain_operator_v1.json").read_text(
            encoding="utf-8"
        )
    )
    positive = json.loads(
        (ROOT / "configs/u5_r2j0_positive_film_response_v1.json").read_text(
            encoding="utf-8"
        )
    )
    functions: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for name, target in config["targets"].items():
        family = target["family"]
        strength = float(target["strength"])
        if family == "identity":
            functions[name] = lambda rgb: rgb.copy()
        elif family == "u5_r2e0_density":
            operator = operator_from_config(
                density["witnesses"][target["witness"]]
            )
            functions[name] = (
                lambda rgb, op=operator, value=strength: op.apply(
                    rgb, strength=value
                )
            )
        elif family == "u5_r2j0_positive":
            operator = positive_film_operator_from_config(
                positive["witnesses"][target["witness"]]
            )
            functions[name] = (
                lambda rgb, op=operator, value=strength: op.apply(
                    rgb, strength=value
                )
            )
        else:
            raise ValueError(f"unsupported target family: {family}")
    return functions


def _best_global_affine(
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    confirm_source: np.ndarray,
) -> np.ndarray:
    fit_design = np.column_stack((np.ones(len(fit_source)), fit_source))
    coefficients = np.linalg.lstsq(fit_design, fit_target, rcond=None)[0]
    confirm_design = np.column_stack((np.ones(len(confirm_source)), confirm_source))
    return confirm_design @ coefficients


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def run_audit(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    candidate = config["candidate"]
    fit = config["fit"]
    gates = config["gates"]
    fit_source = _grid(int(fit["grid_axis_size"]))
    confirm_source = _grid(int(config["confirmation_grid_axis_size"]))
    jacobian_axis = np.linspace(
        0.05, 0.95, int(config["jacobian_grid_axis_size"]), dtype=np.float64
    )
    jacobian_points = np.stack(
        np.meshgrid(jacobian_axis, jacobian_axis, jacobian_axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    functions = _target_functions(config)

    reports: dict[str, Any] = {}
    for target_name, function in functions.items():
        fit_target = function(fit_source)
        confirm_target = function(confirm_source)
        operator = fit_triangular_monotone_coupling(
            fit_source,
            fit_target,
            stage_channels=candidate["stage_channels"],
            axis_size=int(candidate["gaussian_axis_size"]),
            sigma=float(candidate["isotropic_sigma"]),
            epsilon=float(candidate["epsilon"]),
            maximum_absolute_coefficient=float(
                candidate["maximum_absolute_coefficient"]
            ),
            seed=int(fit["seed"]),
            steps=int(fit["steps"]),
            learning_rate=float(fit["learning_rate"]),
            coefficient_l2=float(fit["coefficient_l2"]),
            gradient_clip_norm=float(fit["gradient_clip_norm"]),
            thread_count=int(fit["thread_count"]),
        )
        prediction = operator.apply(confirm_source)
        affine = _best_global_affine(fit_source, fit_target, confirm_source)
        affine_rmse = _rmse(affine, confirm_target)
        confirmation_rmse = _rmse(prediction, confirm_target)
        jacobians = finite_difference_jacobians(
            operator,
            jacobian_points,
            step=float(config["finite_difference_step"]),
        )
        determinants = np.linalg.det(jacobians)
        spectral_norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
        replay = TriangularMonotoneCouplingOperator.from_dict(
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
        restored = operator.inverse(prediction)
        reports[target_name] = {
            "stage_count": len(operator.stage_channels),
            "parameter_count": int(operator.coefficients.size),
            "maximum_absolute_coefficient": float(
                np.max(np.abs(operator.coefficients))
            ),
            "fit_rgb_rmse": _rmse(operator.apply(fit_source), fit_target),
            "global_affine_confirmation_rgb_rmse": affine_rmse,
            "confirmation_rgb_rmse": confirmation_rmse,
            "confirmation_maximum_absolute_error": float(
                np.max(np.abs(prediction - confirm_target))
            ),
            "improvement_over_global_affine_fraction": (
                float((affine_rmse - confirmation_rmse) / affine_rmse)
                if affine_rmse > 0.0
                else 0.0
            ),
            "output_minimum": float(np.min(prediction)),
            "output_maximum": float(np.max(prediction)),
            "minimum_analytic_stage_derivative": operator.minimum_stage_derivative(
                jacobian_points
            ),
            "minimum_finite_difference_jacobian_determinant": float(
                np.min(determinants)
            ),
            "maximum_finite_difference_jacobian_spectral_norm": float(
                np.max(spectral_norms)
            ),
            "inverse_roundtrip_maximum_absolute_error": float(
                np.max(np.abs(restored - confirm_source))
            ),
            "serialization_replay_maximum_absolute_error": float(
                np.max(np.abs(replay.apply(confirm_source) - prediction))
            ),
            "partition_maximum_absolute_error": float(
                np.max(np.abs(partitioned - prediction))
            ),
        }

    checks = {
        "identity_exact": (
            reports["identity"]["confirmation_maximum_absolute_error"]
            <= gates["identity_confirmation_maximum_absolute_error"]
        ),
        "bounded": all(
            report["output_minimum"] >= gates["output_minimum"]
            and report["output_maximum"] <= gates["output_maximum"]
            for report in reports.values()
        ),
        "coefficient_bound": all(
            report["maximum_absolute_coefficient"]
            <= gates["maximum_absolute_coefficient"]
            for report in reports.values()
        ),
        "confirmation_fidelity": all(
            report["confirmation_rgb_rmse"]
            <= gates["maximum_confirmation_rgb_rmse"]
            for report in reports.values()
        ),
        "nonlinear_gain": all(
            report["improvement_over_global_affine_fraction"]
            >= gates["minimum_improvement_over_global_affine_fraction"]
            for name, report in reports.items()
            if name != "identity"
        ),
        "analytic_stage_derivative": all(
            report["minimum_analytic_stage_derivative"]
            >= gates["minimum_analytic_stage_derivative"]
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
        "inverse_roundtrip": all(
            report["inverse_roundtrip_maximum_absolute_error"]
            <= gates["inverse_roundtrip_maximum_absolute_error"]
            for report in reports.values()
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
        "fit_grid_points": len(fit_source),
        "confirmation_grid_points": len(confirm_source),
        "jacobian_grid_points": len(jacobian_points),
        "targets": reports,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2k1_triangular_monotone_coupling_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2k1_triangular_monotone_coupling_v1/report.json",
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
