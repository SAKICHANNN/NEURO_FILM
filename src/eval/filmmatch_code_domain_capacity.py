"""Grouped capacity audit for the FilmMatch paired code-domain source."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_paired_source import canonical_sha256
from src.real_film.velvia_chart_explainability import _fit_affine
from src.roll2film.emulating_emulsion_baseline import (
    EmulatingEmulsionFitResult,
    fit_emulating_emulsion_equation,
)
from src.roll2film.monotone_curve_matrix import (
    MonotoneCurveMatrixFitResult,
    fit_monotone_curve_positive_matrix,
)
from src.roll2film.positive_film_fitting import (
    PositiveFilmFitResult,
    fit_positive_film_response_operator,
)


MODEL_NAMES = (
    "identity",
    "full_affine",
    "bounded_positive_one_matrix",
    "bounded_positive_two_matrix",
    "monotone_curves_positive_matrix",
    "emulating_emulsion_equation_30p",
)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f8").tobytes()
    ).hexdigest()


def prediction_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    rendered = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if (
        rendered.ndim != 2
        or rendered.shape != reference.shape
        or rendered.shape[1] != 3
        or not np.all(np.isfinite(rendered))
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError("metrics require matching finite Nx3 arrays")
    error = rendered - reference
    euclidean = np.linalg.norm(error, axis=1)
    return {
        "rgb_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_rgb_euclidean": float(np.median(euclidean)),
        "p95_rgb_euclidean": float(np.quantile(euclidean, 0.95)),
        "maximum_rgb_euclidean": float(np.max(euclidean)),
        "out_of_cube_sample_fraction": float(
            np.mean(np.any((rendered < 0.0) | (rendered > 1.0), axis=1))
        ),
    }


def _fit_options(config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    fit = config["development"]["fit"]
    phase_fit = fit[phase]
    return {
        "restart_count": int(phase_fit["restart_count"]),
        "maximum_function_evaluations": int(
            phase_fit["maximum_function_evaluations"]
        ),
        "function_tolerance": float(fit["function_tolerance"]),
        "parameter_tolerance": float(fit["parameter_tolerance"]),
        "gradient_tolerance": float(fit["gradient_tolerance"]),
        "seed": int(fit["seed"]),
    }


def fit_models(
    source: np.ndarray,
    target: np.ndarray,
    *,
    config: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    options = _fit_options(config, phase)
    parameterization = config["development"]["parameterization"]
    _, affine = _fit_affine(source, target, per_channel=False)
    bounded = parameterization["bounded_positive"]
    bounded_options = {
        **options,
        "identity_mixture": float(bounded["identity_mixture"]),
        "loss": str(config["development"]["fit"]["bounded_loss"]),
        "loss_scale": float(
            config["development"]["fit"]["bounded_loss_scale"]
        ),
    }
    curve = parameterization["monotone_curves_positive_matrix"]
    emulsion = parameterization["emulating_emulsion_equation_30p"]
    return {
        "full_affine": affine,
        "bounded_positive_one_matrix": fit_positive_film_response_operator(
            source,
            target,
            model="one_matrix",
            **bounded_options,
        ),
        "bounded_positive_two_matrix": fit_positive_film_response_operator(
            source,
            target,
            model="two_matrix",
            **bounded_options,
        ),
        "monotone_curves_positive_matrix": fit_monotone_curve_positive_matrix(
            source,
            target,
            curve_identity_mixture=float(curve["curve_identity_mixture"]),
            matrix_identity_mixture=float(curve["matrix_identity_mixture"]),
            free_logit_bounds=tuple(map(float, curve["free_logit_bounds"])),
            loss=str(config["development"]["fit"]["bounded_loss"]),
            loss_scale=float(
                config["development"]["fit"]["bounded_loss_scale"]
            ),
            **options,
        ),
        "emulating_emulsion_equation_30p": fit_emulating_emulsion_equation(
            source,
            target,
            capture_matrix_entry_bounds=tuple(
                map(float, emulsion["capture_matrix_entry_bounds"])
            ),
            scan_matrix_entry_bounds=tuple(
                map(float, emulsion["scan_matrix_entry_bounds"])
            ),
            response_amplitude_bounds=tuple(
                map(float, emulsion["response_amplitude_bounds"])
            ),
            response_slope_bounds=tuple(
                map(float, emulsion["response_slope_bounds"])
            ),
            response_midpoint_bounds=tuple(
                map(float, emulsion["response_midpoint_bounds"])
            ),
            response_offset_bounds=tuple(
                map(float, emulsion["response_offset_bounds"])
            ),
            **options,
        ),
    }


def apply_models(source: np.ndarray, fits: Mapping[str, Any]) -> dict[str, np.ndarray]:
    affine = fits["full_affine"]
    return {
        "identity": np.asarray(source, dtype=np.float64),
        "full_affine": (
            source @ np.asarray(affine["matrix"], dtype=np.float64).T
            + np.asarray(affine["bias"], dtype=np.float64)
        ),
        **{
            name: fits[name].operator.apply(source)
            for name in MODEL_NAMES
            if name not in {"identity", "full_affine"}
        },
    }


def _fit_payload(fits: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"full_affine": fits["full_affine"]}
    for name in MODEL_NAMES:
        if name in {"identity", "full_affine"}:
            continue
        fit = fits[name]
        output[name] = {
            "converged": bool(fit.converged),
            "development_rgb_rmse": float(fit.development_rgb_rmse),
            "development_maximum_absolute_error": float(
                fit.development_maximum_absolute_error
            ),
            "function_evaluations": int(fit.function_evaluations),
            "restart_index": int(fit.restart_index),
            "operator": fit.operator.to_dict(),
        }
    return output


def _cube(size: int, *, margin: float = 0.0) -> np.ndarray:
    if margin < 0.0 or margin >= 0.5:
        raise ValueError("cube margin must be in [0, 0.5)")
    axis = np.linspace(margin, 1.0 - margin, size, dtype=np.float64)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _finite_difference_determinants(
    operator: Any, cube: np.ndarray, *, step: float = 1e-5
) -> np.ndarray:
    jacobian = np.empty((len(cube), 3, 3), dtype=np.float64)
    for channel in range(3):
        lower = cube.copy()
        upper = cube.copy()
        lower[:, channel] = np.maximum(lower[:, channel] - step, 0.0)
        upper[:, channel] = np.minimum(upper[:, channel] + step, 1.0)
        denominator = upper[:, channel] - lower[:, channel]
        jacobian[:, :, channel] = (
            operator.apply(upper) - operator.apply(lower)
        ) / denominator[:, None]
    return np.linalg.det(jacobian)


def _structural_audit(
    fits: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    controls = config["development"]["structural_audit"]
    cube = _cube(int(controls["cube_size"]))
    jacobian_cube = _cube(
        int(controls["cube_size"]),
        margin=float(controls["jacobian_interior_margin"]),
    )
    predictions = apply_models(cube, fits)
    output: dict[str, dict[str, Any]] = {}
    for name, prediction in predictions.items():
        row: dict[str, Any] = {
            "finite": bool(np.all(np.isfinite(prediction))),
            "minimum_output": float(np.min(prediction)),
            "maximum_output": float(np.max(prediction)),
            "out_of_cube_sample_fraction": float(
                np.mean(np.any((prediction < 0.0) | (prediction > 1.0), axis=1))
            ),
        }
        if name == "identity":
            determinants = np.ones(len(jacobian_cube), dtype=np.float64)
        elif name == "full_affine":
            determinant = np.linalg.det(
                np.asarray(fits[name]["matrix"], dtype=np.float64)
            )
            determinants = np.full(
                len(jacobian_cube), determinant, dtype=np.float64
            )
        else:
            operator = fits[name].operator
            if hasattr(operator, "jacobian_determinants"):
                determinants = operator.jacobian_determinants(jacobian_cube)
            else:
                determinants = _finite_difference_determinants(
                    operator, jacobian_cube
                )
        row["minimum_jacobian_determinant"] = float(np.min(determinants))
        row["maximum_jacobian_determinant"] = float(np.max(determinants))
        row["jacobian_interior_margin"] = float(
            controls["jacobian_interior_margin"]
        )
        row["safe"] = bool(
            row["finite"]
            and row["out_of_cube_sample_fraction"]
            <= float(controls["maximum_safe_out_of_cube_fraction"])
            and row["minimum_jacobian_determinant"]
            >= float(controls["minimum_jacobian_determinant"])
        )
        output[name] = row
    return output


def _fold(
    *,
    fold_id: str,
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    held_source: np.ndarray,
    held_target: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fits = fit_models(fit_source, fit_target, config=config, phase="cross_validation")
    predictions = apply_models(held_source, fits)
    return {
        "fold_id": fold_id,
        "development_samples": int(len(fit_source)),
        "held_samples": int(len(held_source)),
        "metrics": {
            name: prediction_metrics(prediction, held_target)
            for name, prediction in predictions.items()
        },
        "fit": _fit_payload(fits),
    }


def _aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in MODEL_NAMES:
        rmse = np.asarray(
            [fold["metrics"][name]["rgb_rmse"] for fold in folds],
            dtype=np.float64,
        )
        oog = np.asarray(
            [
                fold["metrics"][name]["out_of_cube_sample_fraction"]
                for fold in folds
            ],
            dtype=np.float64,
        )
        output[name] = {
            "folds": len(folds),
            "median_rgb_rmse": float(np.median(rmse)),
            "mean_rgb_rmse": float(np.mean(rmse)),
            "worst_rgb_rmse": float(np.max(rmse)),
            "worst_out_of_cube_sample_fraction": float(np.max(oog)),
        }
    identity = output["identity"]["median_rgb_rmse"]
    affine = output["full_affine"]["median_rgb_rmse"]
    for name in MODEL_NAMES:
        current = output[name]["median_rgb_rmse"]
        output[name]["median_rmse_improvement_over_identity"] = float(
            1.0 - current / identity
        )
        output[name]["median_rmse_improvement_over_affine"] = float(
            1.0 - current / affine
        )
    return output


def select_development_candidate(
    reflective: Mapping[str, Any],
    hue: Mapping[str, Any],
    structural: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = [
        name
        for name in MODEL_NAMES
        if name != "identity"
        and bool(structural[name]["safe"])
        and reflective[name]["worst_out_of_cube_sample_fraction"] == 0.0
        and hue[name]["worst_out_of_cube_sample_fraction"] == 0.0
    ]
    if not eligible:
        return {"selected": None, "eligible": [], "reason": "no safe candidate"}
    complexity = {name: index for index, name in enumerate(MODEL_NAMES)}
    score = {
        name: float(
            reflective[name]["mean_rgb_rmse"] + hue[name]["mean_rgb_rmse"]
        )
        for name in eligible
    }
    best_score = min(score.values())
    near_best = [
        name for name in eligible if score[name] <= best_score * 1.02
    ]
    selected = min(near_best, key=lambda name: complexity[name])
    return {
        "selected": selected,
        "eligible": eligible,
        "combined_mean_rmse_score": score,
        "near_best_within_two_percent": near_best,
        "reason": "lowest grouped error with a two-percent simplest-model tie break",
    }


def evaluate_capacity(
    datasets: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reflective_source = np.asarray(datasets["reflective_source"], dtype=np.float64)
    reflective_target = np.asarray(datasets["reflective_target"], dtype=np.float64)
    emissive_source = np.asarray(datasets["emissive_source"], dtype=np.float64)
    emissive_target = np.asarray(datasets["emissive_target"], dtype=np.float64)
    reflective_records = datasets["reflective_records"]
    emissive_records = datasets["emissive_records"]

    illuminants = np.asarray(
        [record["illuminant"] for record in reflective_records]
    )
    reflective_folds = []
    for illuminant in ("5600K", "3200K", "5600K_CTB"):
        held = illuminants == illuminant
        reflective_folds.append(
            _fold(
                fold_id=f"held-illuminant-{illuminant}",
                fit_source=reflective_source[~held],
                fit_target=reflective_target[~held],
                held_source=reflective_source[held],
                held_target=reflective_target[held],
                config=config,
            )
        )

    hue_labels = np.asarray(
        [record["hue_sector"] for record in emissive_records]
    )
    hue_folds = []
    for hue in sorted(set(hue_labels)):
        held = hue_labels == hue
        hue_folds.append(
            _fold(
                fold_id=f"held-emissive-hue-{hue}",
                fit_source=np.concatenate(
                    [reflective_source, emissive_source[~held]]
                ),
                fit_target=np.concatenate(
                    [reflective_target, emissive_target[~held]]
                ),
                held_source=emissive_source[held],
                held_target=emissive_target[held],
                config=config,
            )
        )

    combined_source = np.concatenate([reflective_source, emissive_source])
    combined_target = np.concatenate([reflective_target, emissive_target])
    final_fits = fit_models(
        combined_source, combined_target, config=config, phase="final"
    )
    structural = _structural_audit(final_fits, config=config)
    reflective_aggregate = _aggregate(reflective_folds)
    hue_aggregate = _aggregate(hue_folds)
    selection = select_development_candidate(
        reflective_aggregate, hue_aggregate, structural
    )
    readout = config["development"]["development_readout"]
    selected = selection["selected"]
    readout_passed = bool(
        selected is not None
        and reflective_aggregate[selected][
            "median_rmse_improvement_over_identity"
        ]
        >= float(
            readout[
                "minimum_median_reflective_rmse_improvement_over_identity"
            ]
        )
        and reflective_aggregate[selected][
            "median_rmse_improvement_over_affine"
        ]
        >= float(
            readout[
                "minimum_median_reflective_rmse_improvement_over_affine"
            ]
        )
        and hue_aggregate[selected]["median_rmse_improvement_over_identity"]
        >= float(
            readout["minimum_median_hue_rmse_improvement_over_identity"]
        )
    )
    report = {
        "schema_version": "u5-r2aw1-filmmatch-code-domain-capacity-report-v1",
        "experiment_id": config["experiment_id"],
        "input_array_sha256": {
            name: _array_sha256(np.asarray(datasets[name]))
            for name in (
                "reflective_source",
                "reflective_target",
                "emissive_source",
                "emissive_target",
            )
        },
        "reflective_folds": reflective_folds,
        "emissive_hue_folds": hue_folds,
        "reflective_aggregate": reflective_aggregate,
        "emissive_hue_aggregate": hue_aggregate,
        "final_fit": _fit_payload(final_fits),
        "structural_audit": structural,
        "capacity_leader": selection,
        "development_champion": selected if readout_passed else None,
        "development_readout_passed": readout_passed,
        "validation_required": bool(
            readout["validation_required_before_any_promotion"]
        ),
        "promotion_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = [
    "MODEL_NAMES",
    "apply_models",
    "evaluate_capacity",
    "fit_models",
    "prediction_metrics",
    "select_development_candidate",
]
