"""Cross-domain capacity audit for the clean-room curve-then-matrix baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.real_film.velvia_chart_explainability import _fit_affine, _metrics
from src.real_film.velvia_cross_domain import load_cross_domain_pairs
from src.roll2film.monotone_curve_matrix import (
    MonotoneCurveMatrixFitResult,
    fit_monotone_curve_positive_matrix,
)
from src.roll2film.positive_film_fitting import (
    PositiveFilmFitResult,
    fit_positive_film_response_operator,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def load_ao4m_pairs(
    chart_path: Path,
    palette_path: Path,
    loader_config_path: Path,
    config: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    source = config["paired_source"]
    raw = loader_config_path.read_bytes()
    if _sha256(raw) != source["loader_config_sha256"]:
        raise ValueError("AO4M loader config hash mismatch")
    pairs = load_cross_domain_pairs(chart_path, palette_path, json.loads(raw))
    return {
        "velvia_chart": pairs["velvia_chart"],
        "velvia_palette": pairs["velvia_palette"],
    }


def _candidate_options(config: dict[str, Any]) -> dict[str, Any]:
    parameterization = config["models"]["candidate_parameterization"]
    fit = config["models"]["fit"]
    curve_bounds = tuple(
        float(v) for v in parameterization["curve_free_logit_bounds"]
    )
    matrix_bounds = tuple(
        float(v) for v in parameterization["matrix_free_logit_bounds"]
    )
    if curve_bounds != matrix_bounds:
        raise ValueError("AO4M v1 requires one frozen free-logit bound")
    return {
        "curve_identity_mixture": float(
            parameterization["curve_identity_mixture"]
        ),
        "matrix_identity_mixture": float(
            parameterization["matrix_identity_mixture"]
        ),
        "free_logit_bounds": curve_bounds,
        "restart_count": int(fit["restart_count"]),
        "maximum_function_evaluations": int(
            fit["maximum_function_evaluations"]
        ),
        "function_tolerance": float(fit["function_tolerance"]),
        "parameter_tolerance": float(fit["parameter_tolerance"]),
        "gradient_tolerance": float(fit["gradient_tolerance"]),
        "loss": str(fit["loss"]),
        "seed": int(fit["seed"]),
    }


def _bounded_options(config: dict[str, Any]) -> dict[str, Any]:
    fit = config["models"]["bounded_control_fit"]
    return {
        "identity_mixture": float(fit["identity_mixture"]),
        "restart_count": int(fit["restart_count"]),
        "maximum_function_evaluations": int(
            fit["maximum_function_evaluations"]
        ),
        "function_tolerance": float(fit["function_tolerance"]),
        "parameter_tolerance": float(fit["parameter_tolerance"]),
        "gradient_tolerance": float(fit["gradient_tolerance"]),
        "loss": str(fit["loss"]),
        "loss_scale": float(fit["loss_scale"]),
        "seed": int(fit["seed"]),
    }


def _fit_all(
    source: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    _, affine = _fit_affine(source, target, per_channel=False)
    return {
        "full_affine": affine,
        "bounded_one_matrix": fit_positive_film_response_operator(
            source,
            target,
            model="one_matrix",
            **_bounded_options(config),
        ),
        "candidate": fit_monotone_curve_positive_matrix(
            source,
            target,
            **_candidate_options(config),
        ),
    }


def _predict_all(source: np.ndarray, fits: dict[str, Any]) -> dict[str, np.ndarray]:
    affine = fits["full_affine"]
    one: PositiveFilmFitResult = fits["bounded_one_matrix"]
    candidate: MonotoneCurveMatrixFitResult = fits["candidate"]
    return {
        "identity": source,
        "full_affine": (
            source @ np.asarray(affine["matrix"], dtype=np.float64).T
            + np.asarray(affine["bias"], dtype=np.float64)
        ),
        "bounded_one_matrix": one.operator.apply(source),
        "candidate": candidate.operator.apply(source),
    }


def _fit_record(fits: dict[str, Any]) -> dict[str, Any]:
    one: PositiveFilmFitResult = fits["bounded_one_matrix"]
    candidate: MonotoneCurveMatrixFitResult = fits["candidate"]
    candidate_payload = candidate.operator.to_dict()
    return {
        "full_affine": fits["full_affine"],
        "bounded_one_matrix": {
            "converged": one.converged,
            "development_rgb_rmse": one.development_rgb_rmse,
            "restart_index": one.restart_index,
            "operator": one.operator.to_dict(),
        },
        "candidate": {
            "converged": candidate.converged,
            "development_rgb_rmse": candidate.development_rgb_rmse,
            "development_maximum_absolute_error": (
                candidate.development_maximum_absolute_error
            ),
            "optimization_cost": candidate.optimization_cost,
            "optimality": candidate.optimality,
            "function_evaluations": candidate.function_evaluations,
            "restart_index": candidate.restart_index,
            "operator_sha256": _sha256(_canonical_json(candidate_payload)),
            "operator": candidate_payload,
        },
    }


def _cube(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )


def _candidate_structure(
    candidate: MonotoneCurveMatrixFitResult, cube: np.ndarray
) -> dict[str, float | bool]:
    operator = candidate.operator
    endpoints = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    endpoint_error = float(np.max(np.abs(operator.apply(endpoints) - endpoints)))
    output = operator.apply(cube)
    inverse_error = float(np.max(np.abs(operator.inverse(output) - cube)))
    determinants = operator.jacobian_determinants(cube)
    return {
        "endpoint_maximum_absolute_error": endpoint_error,
        "inverse_roundtrip_maximum_absolute_error": inverse_error,
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "maximum_jacobian_determinant": float(np.max(determinants)),
        "cube_in_range": bool(np.all(output >= 0.0) and np.all(output <= 1.0)),
    }


def evaluate_curve_matrix_capacity(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if set(datasets) != {"velvia_chart", "velvia_palette"}:
        raise ValueError("AO4M requires exact chart and palette domains")
    counts = {"velvia_chart": 24, "velvia_palette": 47}
    for name, count in counts.items():
        source, target = datasets[name]
        if (
            source.shape != (count, 3)
            or target.shape != source.shape
            or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(target))
        ):
            raise ValueError(f"AO4M {name} data shape or finiteness mismatch")

    originals = {
        name: (source.copy(), target.copy())
        for name, (source, target) in datasets.items()
    }
    names = ("identity", "full_affine", "bounded_one_matrix", "candidate")
    fold_count = int(config["evaluation"]["cross_validation"]["fold_count"])
    predictions = {
        name: {
            domain: np.full_like(datasets[domain][1], np.nan, dtype=np.float64)
            for domain in counts
        }
        for name in names
    }
    folds: list[dict[str, Any]] = []
    fit_failures: list[dict[str, Any]] = []
    candidate_development_rmse: list[float] = []
    candidate_fold_oog: list[float] = []
    structures: list[dict[str, float | bool]] = []
    cube = _cube(int(config["evaluation"]["jacobian_grid_size"]))

    for fold in range(fold_count):
        development_source: list[np.ndarray] = []
        development_target: list[np.ndarray] = []
        held: dict[str, np.ndarray] = {}
        for domain in counts:
            source, target = datasets[domain]
            indices = np.arange(len(source))
            held_indices = indices[indices % fold_count == fold]
            development_indices = indices[indices % fold_count != fold]
            held[domain] = held_indices
            development_source.append(source[development_indices])
            development_target.append(target[development_indices])
        try:
            fitted = _fit_all(
                np.concatenate(development_source),
                np.concatenate(development_target),
                config,
            )
            candidate: MonotoneCurveMatrixFitResult = fitted["candidate"]
            candidate_development_rmse.append(candidate.development_rgb_rmse)
            structure = _candidate_structure(candidate, cube)
            structures.append(structure)
            fold_metrics: dict[str, Any] = {}
            for domain in counts:
                source, target = datasets[domain]
                held_indices = held[domain]
                domain_predictions = _predict_all(source[held_indices], fitted)
                fold_metrics[domain] = {}
                for name, prediction in domain_predictions.items():
                    predictions[name][domain][held_indices] = prediction
                    fold_metrics[domain][name] = _metrics(
                        prediction, target[held_indices]
                    )
                candidate_fold_oog.append(
                    fold_metrics[domain]["candidate"][
                        "raw_out_of_cube_fraction"
                    ]
                )
            folds.append(
                {
                    "fold": fold,
                    "status": "complete",
                    "held_indices": {
                        domain: indices.tolist()
                        for domain, indices in held.items()
                    },
                    "fit": _fit_record(fitted),
                    "candidate_structure": structure,
                    "confirmation": fold_metrics,
                }
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            fit_failures.append(
                {
                    "scope": "cross_validation",
                    "fold": fold,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            folds.append(
                {"fold": fold, "status": "fit_failure", "message": str(exc)}
            )

    aggregate: dict[str, Any] = {}
    all_predictions_complete = True
    for name in names:
        aggregate[name] = {"domains": {}}
        combined_prediction: list[np.ndarray] = []
        combined_target: list[np.ndarray] = []
        for domain in counts:
            prediction = predictions[name][domain]
            target = datasets[domain][1]
            if not np.all(np.isfinite(prediction)):
                all_predictions_complete = False
                aggregate[name]["domains"][domain] = None
                continue
            aggregate[name]["domains"][domain] = _metrics(prediction, target)
            combined_prediction.append(prediction)
            combined_target.append(target)
        aggregate[name]["combined"] = (
            _metrics(
                np.concatenate(combined_prediction),
                np.concatenate(combined_target),
            )
            if len(combined_prediction) == len(counts)
            else None
        )

    leave_domain_out: list[dict[str, Any]] = []
    for direction in config["evaluation"]["leave_domain_out_directions"]:
        fit_name = str(direction["fit"])
        confirm_name = str(direction["confirm"])
        fit_source, fit_target = datasets[fit_name]
        confirm_source, confirm_target = datasets[confirm_name]
        try:
            fitted = _fit_all(fit_source, fit_target, config)
            direction_predictions = _predict_all(confirm_source, fitted)
            direction_structure = _candidate_structure(
                fitted["candidate"], cube
            )
            structures.append(direction_structure)
            metrics = {
                name: _metrics(prediction, confirm_target)
                for name, prediction in direction_predictions.items()
            }
            candidate_rmse = metrics["candidate"]["rgb_rmse"]
            control_rmse = metrics["bounded_one_matrix"]["rgb_rmse"]
            leave_domain_out.append(
                {
                    "fit": fit_name,
                    "confirm": confirm_name,
                    "status": "complete",
                    "fit_record": _fit_record(fitted),
                    "candidate_structure": direction_structure,
                    "confirmation": metrics,
                    "candidate_rgb_rmse_gain_over_bounded_one_matrix": (
                        1.0 - candidate_rmse / control_rmse
                    ),
                }
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            fit_failures.append(
                {
                    "scope": "leave_domain_out",
                    "fit": fit_name,
                    "confirm": confirm_name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            leave_domain_out.append(
                {
                    "fit": fit_name,
                    "confirm": confirm_name,
                    "status": "fit_failure",
                    "message": str(exc),
                }
            )

    candidate_metrics = aggregate["candidate"]["combined"]
    control_metrics = aggregate["bounded_one_matrix"]["combined"]
    if candidate_metrics is not None and control_metrics is not None:
        rgb_gain = (
            1.0 - candidate_metrics["rgb_rmse"] / control_metrics["rgb_rmse"]
        )
        delta_gain = (
            1.0
            - candidate_metrics["mean_delta_e76"]
            / control_metrics["mean_delta_e76"]
        )
        per_domain_gain = {
            domain: (
                1.0
                - aggregate["candidate"]["domains"][domain]["rgb_rmse"]
                / aggregate["bounded_one_matrix"]["domains"][domain][
                    "rgb_rmse"
                ]
            )
            for domain in counts
        }
        train_test_gap = candidate_metrics["rgb_rmse"] - float(
            np.mean(candidate_development_rmse)
        )
    else:
        rgb_gain = float("-inf")
        delta_gain = float("-inf")
        per_domain_gain = {domain: float("-inf") for domain in counts}
        train_test_gap = float("inf")

    source_nonmutation = all(
        np.array_equal(datasets[name][0], originals[name][0])
        and np.array_equal(datasets[name][1], originals[name][1])
        for name in counts
    )
    ldo_gains = [
        row["candidate_rgb_rmse_gain_over_bounded_one_matrix"]
        for row in leave_domain_out
        if row["status"] == "complete"
    ]
    gates = config["gates"]
    checks = [
        {
            "name": "all_fits_complete",
            "passed": not fit_failures
            and all_predictions_complete
            and len(folds) == fold_count
            and len(ldo_gains) == len(leave_domain_out),
        },
        {
            "name": "equal_candidate_and_control_parameter_count",
            "passed": config["models"]["candidate_parameter_count"]
            == config["models"]["control_parameter_count"]
            == 12,
        },
        {
            "name": "candidate_rgb_rmse_gain",
            "passed": rgb_gain
            >= float(
                gates[
                    "minimum_candidate_rgb_rmse_gain_over_bounded_one_matrix"
                ]
            ),
        },
        {
            "name": "candidate_mean_delta_e76_gain",
            "passed": delta_gain
            >= float(
                gates[
                    "minimum_candidate_mean_delta_e76_gain_over_bounded_one_matrix"
                ]
            ),
        },
        {
            "name": "candidate_rgb_rmse_gain_each_domain",
            "passed": min(per_domain_gain.values())
            >= float(gates["minimum_candidate_rgb_rmse_gain_each_domain"]),
        },
        {
            "name": "candidate_leave_domain_out_rgb_rmse_gain_each_direction",
            "passed": len(ldo_gains) == len(leave_domain_out)
            and min(ldo_gains)
            >= float(
                gates[
                    "minimum_candidate_leave_domain_out_rgb_rmse_gain_each_direction"
                ]
            ),
        },
        {
            "name": "candidate_train_test_gap",
            "passed": train_test_gap
            <= float(gates["maximum_mean_train_test_rgb_rmse_gap"]),
        },
        {
            "name": "candidate_mean_raw_out_of_cube_fraction",
            "passed": candidate_metrics is not None
            and candidate_metrics["raw_out_of_cube_fraction"]
            <= float(gates["maximum_mean_raw_out_of_cube_fraction"]),
        },
        {
            "name": "candidate_maximum_fold_raw_out_of_cube_fraction",
            "passed": bool(candidate_fold_oog)
            and max(candidate_fold_oog)
            <= float(gates["maximum_fold_raw_out_of_cube_fraction"]),
        },
        {
            "name": "candidate_minimum_jacobian_determinant",
            "passed": bool(structures)
            and min(float(row["minimum_jacobian_determinant"]) for row in structures)
            > float(gates["minimum_jacobian_determinant"]),
        },
        {
            "name": "candidate_exact_black_white_endpoints",
            "passed": bool(structures)
            and max(
                float(row["endpoint_maximum_absolute_error"])
                for row in structures
            )
            <= 1e-12,
        },
        {
            "name": "candidate_analytic_inverse_roundtrip",
            "passed": bool(structures)
            and max(
                float(row["inverse_roundtrip_maximum_absolute_error"])
                for row in structures
            )
            <= float(gates["analytic_inverse_roundtrip_maximum"]),
        },
        {"name": "source_nonmutation", "passed": source_nonmutation},
    ]
    automatic_pass = all(bool(check["passed"]) for check in checks)
    return {
        "folds": folds,
        "aggregate": aggregate,
        "leave_domain_out": leave_domain_out,
        "fit_failures": fit_failures,
        "candidate_summary": {
            "rgb_rmse_gain_over_bounded_one_matrix": rgb_gain,
            "mean_delta_e76_gain_over_bounded_one_matrix": delta_gain,
            "rgb_rmse_gain_by_domain": per_domain_gain,
            "mean_train_test_rgb_rmse_gap": train_test_gap,
            "minimum_cube_jacobian_determinant": (
                min(
                    float(row["minimum_jacobian_determinant"])
                    for row in structures
                )
                if structures
                else None
            ),
            "maximum_fold_raw_out_of_cube_fraction": (
                max(candidate_fold_oog) if candidate_fold_oog else None
            ),
            "maximum_endpoint_error": (
                max(
                    float(row["endpoint_maximum_absolute_error"])
                    for row in structures
                )
                if structures
                else None
            ),
            "maximum_inverse_roundtrip_error": (
                max(
                    float(row["inverse_roundtrip_maximum_absolute_error"])
                    for row in structures
                )
                if structures
                else None
            ),
        },
        "automatic_checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            config["decision_if_pass"]
            if automatic_pass
            else config["decision_if_fail"]
        ),
        "source_nonmutation": source_nonmutation,
    }


__all__ = ["evaluate_curve_matrix_capacity", "load_ao4m_pairs"]
