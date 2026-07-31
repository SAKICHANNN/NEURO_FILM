"""Grouped FilmMatch evaluation for the identity-residual sigmoid operator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_code_domain_capacity import prediction_metrics
from src.eval.filmmatch_paired_source import canonical_sha256
from src.roll2film.identity_residual_sigmoid import (
    fit_identity_residual_sigmoid,
)


def _fit(
    source: np.ndarray, target: np.ndarray, config: Mapping[str, Any]
) -> Any:
    candidate = config["candidate"]
    fit = candidate["fit"]
    return fit_identity_residual_sigmoid(
        source,
        target,
        nonlinear_strength=float(candidate["nonlinear_strength"]),
        matrix_identity_mixture=float(candidate["matrix_identity_mixture"]),
        matrix_logit_bounds=tuple(map(float, candidate["matrix_logit_bounds"])),
        midpoint_bounds=tuple(map(float, candidate["midpoint_bounds"])),
        slope_bounds=tuple(map(float, candidate["slope_bounds"])),
        restart_count=int(fit["restart_count"]),
        maximum_function_evaluations=int(fit["maximum_function_evaluations"]),
        function_tolerance=float(fit["function_tolerance"]),
        parameter_tolerance=float(fit["parameter_tolerance"]),
        gradient_tolerance=float(fit["gradient_tolerance"]),
        loss=str(fit["loss"]),
        loss_scale=float(fit["loss_scale"]),
        seed=int(fit["seed"]),
    )


def _evaluate_folds(
    source: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float], list[Any]]:
    rows = []
    fits = []
    for held in sorted(set(labels.tolist())):
        development = labels != held
        validation = ~development
        fit = _fit(source[development], target[development], config)
        prediction = fit.operator.apply(source[validation])
        rows.append(
            {
                "held_group": str(held),
                "development_samples": int(np.count_nonzero(development)),
                "held_samples": int(np.count_nonzero(validation)),
                "converged": bool(fit.converged),
                "metrics": prediction_metrics(prediction, target[validation]),
            }
        )
        fits.append(fit)
    aggregate = {
        "folds": len(rows),
        "mean_rgb_rmse": float(
            np.mean([row["metrics"]["rgb_rmse"] for row in rows])
        ),
        "median_rgb_rmse": float(
            np.median([row["metrics"]["rgb_rmse"] for row in rows])
        ),
        "mean_median_rgb_euclidean": float(
            np.mean(
                [row["metrics"]["median_rgb_euclidean"] for row in rows]
            )
        ),
        "mean_p95_rgb_euclidean": float(
            np.mean([row["metrics"]["p95_rgb_euclidean"] for row in rows])
        ),
        "worst_out_of_cube_sample_fraction": float(
            np.max(
                [
                    row["metrics"]["out_of_cube_sample_fraction"]
                    for row in rows
                ]
            )
        ),
        "all_fits_converged": bool(all(row["converged"] for row in rows)),
    }
    return rows, aggregate, fits


def _cube(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, int(size), dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def evaluate_identity_residual_sigmoid(
    datasets: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    reflective_source = np.asarray(
        datasets["reflective_source"], dtype=np.float64
    )
    reflective_target = np.asarray(
        datasets["reflective_target"], dtype=np.float64
    )
    reflective_labels = np.asarray(
        [str(row["illuminant"]) for row in datasets["reflective_records"]]
    )
    emissive_source = np.asarray(datasets["emissive_source"], dtype=np.float64)
    emissive_target = np.asarray(datasets["emissive_target"], dtype=np.float64)
    emissive_labels = np.asarray(
        [str(row["hue_sector"]) for row in datasets["emissive_records"]]
    )
    reflective_rows, reflective, reflective_fits = _evaluate_folds(
        reflective_source,
        reflective_target,
        reflective_labels,
        config,
    )
    emissive_rows, emissive, emissive_fits = _evaluate_folds(
        emissive_source, emissive_target, emissive_labels, config
    )
    final_fit = _fit(
        np.concatenate((reflective_source, emissive_source)),
        np.concatenate((reflective_target, emissive_target)),
        config,
    )
    cube = _cube(int(config["evaluation"]["cube_size"]))
    audited_fits = reflective_fits + emissive_fits + [final_fit]
    minimum_jacobian = float(
        min(
            np.min(fit.operator.jacobian_determinants(cube))
            for fit in audited_fits
        )
    )
    cube_outputs = [fit.operator.apply(cube) for fit in audited_fits]
    maximum_oog = float(
        max(
            np.mean(np.any((output < 0.0) | (output > 1.0), axis=1))
            for output in cube_outputs
        )
    )
    controls = config["bound_controls"]
    aggregate = {
        "reflective_mean_rgb_rmse": reflective["mean_rgb_rmse"],
        "emissive_mean_rgb_rmse": emissive["mean_rgb_rmse"],
        "combined_mean_rgb_rmse_sum": float(
            reflective["mean_rgb_rmse"] + emissive["mean_rgb_rmse"]
        ),
        "reflective_rmse_ratio_to_original_sigmoid": float(
            reflective["mean_rgb_rmse"]
            / float(
                controls[
                    "original_two_matrix_sigmoid_reflective_mean_rgb_rmse"
                ]
            )
        ),
        "reflective_mean_improvement_over_identity": float(
            1.0
            - reflective["mean_rgb_rmse"]
            / float(controls["identity_reflective_mean_rgb_rmse"])
        ),
        "emissive_rmse_ratio_to_identity": float(
            emissive["mean_rgb_rmse"]
            / float(controls["identity_emissive_mean_rgb_rmse"])
        ),
        "minimum_cube_jacobian_determinant": minimum_jacobian,
        "maximum_out_of_cube_fraction": maximum_oog,
        "all_fits_finite_and_converged": bool(
            reflective["all_fits_converged"]
            and emissive["all_fits_converged"]
            and final_fit.converged
        ),
    }
    gate = config["automatic_gate"]
    gate_results = {
        "reflective_accuracy_retained": aggregate[
            "reflective_rmse_ratio_to_original_sigmoid"
        ]
        <= float(gate["maximum_reflective_rmse_ratio_to_original_sigmoid"]),
        "reflective_value": aggregate[
            "reflective_mean_improvement_over_identity"
        ]
        >= float(gate["minimum_reflective_mean_improvement_over_identity"]),
        "emissive_not_worse_than_identity": aggregate[
            "emissive_rmse_ratio_to_identity"
        ]
        <= float(gate["maximum_emissive_rmse_ratio_to_identity"]),
        "combined_accuracy": aggregate["combined_mean_rgb_rmse_sum"]
        <= float(gate["maximum_combined_mean_rmse_sum"]),
        "out_of_cube": aggregate["maximum_out_of_cube_fraction"]
        <= float(gate["maximum_out_of_cube_fraction"]),
        "jacobian_floor": aggregate["minimum_cube_jacobian_determinant"]
        >= float(gate["minimum_cube_jacobian_determinant"]),
        "finite_converged": aggregate["all_fits_finite_and_converged"],
    }
    passed = bool(all(gate_results.values()))
    if not gate_results["out_of_cube"] or not gate_results["jacobian_floor"]:
        branch = "structural_fail"
    elif passed:
        branch = "pass"
    else:
        branch = "accuracy_fail"
    report = {
        "schema": config["schema"],
        "experiment_id": config["experiment_id"],
        "reflective_folds": reflective_rows,
        "reflective_aggregate": reflective,
        "emissive_folds": emissive_rows,
        "emissive_aggregate": emissive,
        "final_fit": {
            "converged": bool(final_fit.converged),
            "development_rgb_rmse": float(final_fit.development_rgb_rmse),
            "operator": final_fit.operator.to_dict(),
        },
        "aggregate": aggregate,
        "gate_results": gate_results,
        "automatic_gate_passed": passed,
        "branch": branch,
        "validation_comparison_opened": passed,
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = ["evaluate_identity_residual_sigmoid"]
