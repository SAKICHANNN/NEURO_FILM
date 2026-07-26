#!/usr/bin/env python
"""Run the frozen U5.R2K0 bounded Gaussian residual audit."""

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

from src.roll2film.bounded_gaussian_residual import (  # noqa: E402
    BoundedGaussianResidualOperator,
    finite_difference_jacobians,
    fit_bounded_gaussian_residual,
    published_formula_initialization_witness,
)
from src.roll2film.density_domain import operator_from_config  # noqa: E402
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


def _load_target_functions(
    config: dict[str, Any],
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    density_config = json.loads(
        (ROOT / "configs/u5_r2e0_density_domain_operator_v1.json").read_text(
            encoding="utf-8"
        )
    )
    positive_config = json.loads(
        (ROOT / "configs/u5_r2j0_positive_film_response_v1.json").read_text(
            encoding="utf-8"
        )
    )
    result: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for name, target in config["targets"].items():
        family = target["family"]
        strength = float(target["strength"])
        if family == "identity":
            result[name] = lambda rgb: rgb.copy()
        elif family == "u5_r2e0_density":
            operator = operator_from_config(
                density_config["witnesses"][target["witness"]]
            )
            result[name] = (
                lambda rgb, op=operator, value=strength: op.apply(
                    rgb, strength=value
                )
            )
        elif family == "u5_r2j0_positive":
            operator = positive_film_operator_from_config(
                positive_config["witnesses"][target["witness"]]
            )
            result[name] = (
                lambda rgb, op=operator, value=strength: op.apply(
                    rgb, strength=value
                )
            )
        else:
            raise ValueError(f"unsupported target family: {family}")
    return result


def _best_global_affine(
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    confirm_source: np.ndarray,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(fit_source)), fit_source))
    coefficients = np.linalg.lstsq(design, fit_target, rcond=None)[0]
    confirm_design = np.column_stack((np.ones(len(confirm_source)), confirm_source))
    return confirm_design @ coefficients


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def run_audit(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    fit_source = _grid(int(config["fit_grid_axis_size"]))
    confirm_source = _grid(int(config["confirm_grid_axis_size"]))
    jacobian_axis = np.linspace(
        0.05, 0.95, int(config["jacobian_grid_axis_size"]), dtype=np.float64
    )
    jacobian_points = np.stack(
        np.meshgrid(jacobian_axis, jacobian_axis, jacobian_axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    target_functions = _load_target_functions(config)
    gates = config["gates"]

    negative_control_points = np.vstack(
        (fit_source, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    )
    paper_config = config["paper_formula_negative_control"]
    paper_zero = published_formula_initialization_witness(
        negative_control_points,
        axis_size=int(paper_config["gaussian_axis_size"]),
        sigma=float(paper_config["isotropic_sigma"]),
        epsilon=float(paper_config["epsilon"]),
        identity_global=False,
    )
    paper_identity = published_formula_initialization_witness(
        negative_control_points,
        axis_size=int(paper_config["gaussian_axis_size"]),
        sigma=float(paper_config["isotropic_sigma"]),
        epsilon=float(paper_config["epsilon"]),
        identity_global=True,
    )
    paper_metrics = {
        "zero_global_identity_maximum_absolute_error": float(
            np.max(np.abs(paper_zero - negative_control_points))
        ),
        "identity_global_output_maximum_before_clamp": float(np.max(paper_identity)),
        "identity_global_out_of_range_fraction_before_clamp": float(
            np.mean((paper_identity < 0.0) | (paper_identity > 1.0))
        ),
    }

    target_reports: dict[str, Any] = {}
    nonlinear_improvements = []
    for target_name, target_function in target_functions.items():
        fit_target = target_function(fit_source)
        confirm_target = target_function(confirm_source)
        affine_prediction = _best_global_affine(
            fit_source, fit_target, confirm_source
        )
        affine_rmse = _rmse(affine_prediction, confirm_target)
        capacities: dict[str, Any] = {}
        for axis_size in config["candidate"]["gaussian_axis_sizes"]:
            axis_size = int(axis_size)
            sigma = float(config["candidate"]["isotropic_sigmas"][str(axis_size)])
            operator = fit_bounded_gaussian_residual(
                fit_source,
                fit_target,
                axis_size=axis_size,
                sigma=sigma,
                epsilon=float(config["candidate"]["epsilon"]),
                ridge=float(config["candidate"]["ridge"]),
            )
            prediction = operator.apply(confirm_source)
            jacobians = finite_difference_jacobians(
                operator,
                jacobian_points,
                step=float(config["finite_difference_step"]),
            )
            determinants = np.linalg.det(jacobians)
            diagonal = np.diagonal(jacobians, axis1=1, axis2=2)
            spectral_norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
            replay = BoundedGaussianResidualOperator.from_dict(
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
            capacity_report = {
                "primitive_count": int(len(operator.centers)),
                "parameter_count": int(operator.coefficients.size),
                "maximum_absolute_coefficient": float(
                    np.max(np.abs(operator.coefficients))
                ),
                "confirmation_rgb_rmse": _rmse(prediction, confirm_target),
                "confirmation_maximum_absolute_error": float(
                    np.max(np.abs(prediction - confirm_target))
                ),
                "output_minimum": float(np.min(prediction)),
                "output_maximum": float(np.max(prediction)),
                "minimum_finite_difference_jacobian_determinant": float(
                    np.min(determinants)
                ),
                "minimum_finite_difference_diagonal_derivative": float(
                    np.min(diagonal)
                ),
                "maximum_finite_difference_jacobian_spectral_norm": float(
                    np.max(spectral_norms)
                ),
                "serialization_replay_maximum_absolute_error": float(
                    np.max(np.abs(replay.apply(confirm_source) - prediction))
                ),
                "partition_maximum_absolute_error": float(
                    np.max(np.abs(partitioned - prediction))
                ),
                "strength_zero_maximum_absolute_error": float(
                    np.max(
                        np.abs(
                            operator.apply(confirm_source, strength=0.0)
                            - confirm_source
                        )
                    )
                ),
            }
            capacities[f"n{len(operator.centers)}"] = capacity_report
        n27 = capacities["n27"]
        improvement = (
            (affine_rmse - n27["confirmation_rgb_rmse"]) / affine_rmse
            if affine_rmse > 0.0
            else 0.0
        )
        if target_name != "identity":
            nonlinear_improvements.append(float(improvement))
        target_reports[target_name] = {
            "global_affine_confirmation_rgb_rmse": affine_rmse,
            "n27_improvement_over_global_affine_fraction": float(improvement),
            "capacities": capacities,
        }

    all_capacities = [
        capacity
        for target in target_reports.values()
        for capacity in target["capacities"].values()
    ]
    n27_capacities = [
        target["capacities"]["n27"] for target in target_reports.values()
    ]
    checks = {
        "paper_zero_global_is_identity": (
            paper_metrics["zero_global_identity_maximum_absolute_error"]
            <= gates["paper_zero_global_identity_maximum_absolute_error"]
        ),
        "paper_identity_global_exposes_double_count": (
            paper_metrics["identity_global_output_maximum_before_clamp"]
            >= gates["paper_identity_global_minimum_output_maximum"]
        ),
        "identity_exact": (
            target_reports["identity"]["capacities"]["n27"][
                "confirmation_maximum_absolute_error"
            ]
            <= gates["identity_confirmation_maximum_absolute_error"]
        ),
        "bounded": all(
            item["output_minimum"] >= gates["output_minimum"]
            and item["output_maximum"] <= gates["output_maximum"]
            for item in all_capacities
        ),
        "coefficient_bound": all(
            item["maximum_absolute_coefficient"]
            <= gates["maximum_absolute_coefficient"]
            for item in all_capacities
        ),
        "n27_fidelity": all(
            item["confirmation_rgb_rmse"]
            <= gates["maximum_confirmation_rgb_rmse_n27"]
            for item in n27_capacities
        ),
        "n27_nonlinear_gain": all(
            value
            >= gates["minimum_n27_improvement_over_global_affine_fraction"]
            for value in nonlinear_improvements
        ),
        "positive_jacobian": all(
            item["minimum_finite_difference_jacobian_determinant"]
            > gates["minimum_finite_difference_jacobian_determinant"]
            for item in all_capacities
        ),
        "non_negative_diagonal_derivatives": all(
            item["minimum_finite_difference_diagonal_derivative"]
            >= gates["minimum_finite_difference_diagonal_derivative"]
            for item in all_capacities
        ),
        "bounded_jacobian_norm": all(
            item["maximum_finite_difference_jacobian_spectral_norm"]
            <= gates["maximum_finite_difference_jacobian_spectral_norm"]
            for item in all_capacities
        ),
        "serialization_replay": all(
            item["serialization_replay_maximum_absolute_error"]
            <= gates["serialization_replay_maximum_absolute_error"]
            for item in all_capacities
        ),
        "partition_parity": all(
            item["partition_maximum_absolute_error"]
            <= gates["partition_maximum_absolute_error"]
            for item in all_capacities
        ),
        "strength_zero_identity": all(
            item["strength_zero_maximum_absolute_error"] == 0.0
            for item in all_capacities
        ),
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": config_sha256,
        "fit_grid_points": len(fit_source),
        "confirm_grid_points": len(confirm_source),
        "jacobian_grid_points": len(jacobian_points),
        "paper_formula_negative_control": paper_metrics,
        "targets": target_reports,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2k0_bounded_gaussian_residual_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2k0_bounded_gaussian_residual_v1/report.json",
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
