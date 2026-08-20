"""Held-block capacity test for a bounded local-affine bilateral grid."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.fivek_bilateral_gain_capacity import (
    _apply_global_lut,
    _apply_safe_log_gain,
    _difference_matrix,
    _fit_gain,
    _fit_global_lut,
    _gain_features,
    _held_mask,
    _new_boundary_fraction,
    _rgb,
    _rmse,
    _stable_midrank,
)


class FiveKBilateralAffineCapacityError(ValueError):
    """Raised when the frozen affine capacity contract is violated."""


def _affine_design(features: np.ndarray, source: np.ndarray) -> np.ndarray:
    flat = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    augmented = np.concatenate([flat, np.ones((len(flat), 1))], axis=1)
    return (features[:, :, None] * augmented[:, None, :]).reshape(len(flat), -1)


def _fit_affine(
    features: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    spec: Mapping[str, Any],
) -> np.ndarray:
    design = _affine_design(features, source)
    selected = fit_mask.reshape(-1)
    target_delta = (np.asarray(target) - np.asarray(source)).reshape(-1, 3)
    nodes = features.shape[1]
    differences = _difference_matrix(spec["grid_shape"])
    smooth = np.kron(differences, np.eye(4))
    normal = design[selected].T @ design[selected]
    normal += float(spec["ridge"]) * np.eye(nodes * 4)
    normal += float(spec["first_difference_smoothness"]) * (smooth.T @ smooth)
    coefficients = np.linalg.solve(normal, design[selected].T @ target_delta[selected])
    return np.clip(coefficients, -float(spec["coefficient_bound"]), float(spec["coefficient_bound"]))


def _apply_safe_affine(
    source: np.ndarray,
    features: np.ndarray,
    coefficients: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    source_flat = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    delta = _affine_design(features, source) @ coefficients
    epsilon = float(spec["epsilon"])
    dose = np.ones(len(source_flat), dtype=np.float64)
    source_boundary = np.any(
        (source_flat <= epsilon) | (source_flat >= 1.0 - epsilon), axis=1
    )
    dose[source_boundary] = 0.0
    positive = delta > 0.0
    negative = delta < 0.0
    limits = np.full_like(delta, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        limits[positive] = (1.0 - epsilon - source_flat[positive]) / delta[positive]
        limits[negative] = (epsilon - source_flat[negative]) / delta[negative]
    dose = np.clip(np.minimum(dose, np.min(limits, axis=1)), 0.0, 1.0)
    output = source_flat + dose[:, None] * delta
    if (
        not np.all(np.isfinite(output))
        or np.any(output < -1.0e-12)
        or np.any(output > 1.0 + 1.0e-12)
    ):
        raise FiveKBilateralAffineCapacityError("analytical affine guard failed")
    return output.reshape(source.shape), dose.reshape(source.shape[:2])


def _improvement(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float((reference.mean() - candidate.mean()) / reference.mean())


def evaluate_affine_capacity(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    expected = int(config["population"]["source_count_exact"])
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    if len(ordered) != expected or len({str(row["group"]) for row in ordered}) != expected:
        raise FiveKBilateralAffineCapacityError("population identity drift")
    candidate = config["candidate"]
    controls = config["controls"]
    held_spec = config["held_block_protocol"]
    boundary_epsilon = float(config["evaluation"]["boundary_epsilon"])
    output_rows: list[dict[str, Any]] = []
    for row in ordered:
        source_encoded = _rgb(row["source"])
        target_encoded = _rgb(row["target"])
        if source_encoded.shape != target_encoded.shape:
            raise FiveKBilateralAffineCapacityError("paired shape mismatch")
        source_linear = encoded_srgb_to_linear(source_encoded)
        target_linear = encoded_srgb_to_linear(target_encoded)
        luma = np.tensordot(
            source_linear, np.asarray([0.2126, 0.7152, 0.0722]), axes=([-1], [0])
        )
        rank = _stable_midrank(luma)
        candidate_features = _gain_features(source_linear.shape[:2], rank, candidate["grid_shape"])
        luma_spec = {**candidate, **controls["luminance_only_affine"]}
        spatial_spec = {**candidate, **controls["spatial_only_affine"]}
        luma_features = _gain_features(source_linear.shape[:2], rank, luma_spec["grid_shape"])
        spatial_features = _gain_features(source_linear.shape[:2], rank, spatial_spec["grid_shape"])
        bt0_spec = controls["closed_bt0_gain"]
        bt0_features = _gain_features(source_linear.shape[:2], rank, bt0_spec["grid_shape"])
        for fold in map(int, held_spec["folds"]):
            held = _held_mask(source_linear.shape[:2], held_spec, fold)
            fit = ~held
            candidate_coefficients = _fit_affine(
                candidate_features, source_linear, target_linear, fit, candidate
            )
            luma_coefficients = _fit_affine(
                luma_features, source_linear, target_linear, fit, luma_spec
            )
            spatial_coefficients = _fit_affine(
                spatial_features, source_linear, target_linear, fit, spatial_spec
            )
            bt0_coefficients = _fit_gain(
                bt0_features, source_linear, target_linear, fit, bt0_spec
            )
            lut_coefficients = _fit_global_lut(
                source_encoded,
                target_encoded,
                fit,
                controls["parameter_matched_global_rgb_lut"],
                boundary_epsilon,
            )
            candidate_linear, dose = _apply_safe_affine(
                source_linear, candidate_features, candidate_coefficients, candidate
            )
            luma_linear, _ = _apply_safe_affine(
                source_linear, luma_features, luma_coefficients, luma_spec
            )
            spatial_linear, _ = _apply_safe_affine(
                source_linear, spatial_features, spatial_coefficients, spatial_spec
            )
            bt0_linear, _ = _apply_safe_log_gain(
                source_linear, bt0_features, bt0_coefficients, bt0_spec
            )
            candidate_encoded = linear_srgb_to_encoded(candidate_linear)
            luma_encoded = linear_srgb_to_encoded(luma_linear)
            spatial_encoded = linear_srgb_to_encoded(spatial_linear)
            bt0_encoded = linear_srgb_to_encoded(bt0_linear)
            global_encoded = _apply_global_lut(
                source_encoded, lut_coefficients, boundary_epsilon
            )
            identity_rmse = _rmse(source_encoded, target_encoded, held)
            output_rows.append(
                {
                    "pair_id": str(row["pair_id"]),
                    "group": str(row["group"]),
                    "target_variant": str(row["target_variant"]),
                    "fold": fold,
                    "fit_pixels": int(np.sum(fit)),
                    "held_pixels": int(np.sum(held)),
                    "identity_rmse": identity_rmse,
                    "candidate_fit_rmse": _rmse(candidate_encoded, target_encoded, fit),
                    "candidate_rmse": _rmse(candidate_encoded, target_encoded, held),
                    "global_lut_rmse": _rmse(global_encoded, target_encoded, held),
                    "bt0_gain_rmse": _rmse(bt0_encoded, target_encoded, held),
                    "luminance_only_rmse": _rmse(luma_encoded, target_encoded, held),
                    "spatial_only_rmse": _rmse(spatial_encoded, target_encoded, held),
                    "candidate_style_retention": _rmse(candidate_encoded, source_encoded, held)
                    / max(identity_rmse, 1.0e-12),
                    "minimum_affine_dose": float(np.min(dose[held])),
                    "new_boundary_fraction": _new_boundary_fraction(
                        source_encoded, candidate_encoded, held, boundary_epsilon
                    ),
                    "out_of_cube_fraction": float(
                        np.mean((candidate_encoded < 0.0) | (candidate_encoded > 1.0))
                    ),
                }
            )
    candidate_error = np.asarray([row["candidate_rmse"] for row in output_rows])
    global_error = np.asarray([row["global_lut_rmse"] for row in output_rows])
    bt0_error = np.asarray([row["bt0_gain_rmse"] for row in output_rows])
    luma_error = np.asarray([row["luminance_only_rmse"] for row in output_rows])
    spatial_error = np.asarray([row["spatial_only_rmse"] for row in output_rows])
    fit_error = np.asarray([row["candidate_fit_rmse"] for row in output_rows])
    metrics = {
        "rows": len(output_rows),
        "mean_improvement_over_parameter_matched_global_lut": _improvement(global_error, candidate_error),
        "win_fraction_over_parameter_matched_global_lut": float(np.mean(candidate_error < global_error)),
        "p95_error_ratio_to_parameter_matched_global_lut": float(
            np.quantile(candidate_error, 0.95) / np.quantile(global_error, 0.95)
        ),
        "worst_error_ratio_to_parameter_matched_global_lut": float(
            np.max(candidate_error) / np.max(global_error)
        ),
        "mean_improvement_over_closed_bt0_gain": _improvement(bt0_error, candidate_error),
        "win_fraction_over_closed_bt0_gain": float(np.mean(candidate_error < bt0_error)),
        "mean_improvement_over_luminance_only": _improvement(luma_error, candidate_error),
        "mean_improvement_over_spatial_only": _improvement(spatial_error, candidate_error),
        "median_style_retention": float(
            np.median([row["candidate_style_retention"] for row in output_rows])
        ),
        "p95_held_to_fit_error_ratio": float(
            np.quantile(candidate_error / np.maximum(fit_error, 1.0e-12), 0.95)
        ),
        "maximum_new_boundary_fraction": float(
            max(row["new_boundary_fraction"] for row in output_rows)
        ),
        "maximum_out_of_cube_fraction": float(
            max(row["out_of_cube_fraction"] for row in output_rows)
        ),
        "minimum_affine_dose": float(min(row["minimum_affine_dose"] for row in output_rows)),
        "candidate_error_sha256": hashlib.sha256(
            np.ascontiguousarray(candidate_error, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    gates = config["evaluation"]["automatic_gates"]
    checks = {
        "source_count": len(ordered) == int(gates["source_count_exact"]),
        "folds": len({row["fold"] for row in output_rows}) == int(gates["fold_count_exact"]),
        "global_mean": metrics["mean_improvement_over_parameter_matched_global_lut"]
        >= float(gates["minimum_mean_improvement_over_parameter_matched_global_lut"]),
        "global_wins": metrics["win_fraction_over_parameter_matched_global_lut"]
        >= float(gates["minimum_win_fraction_over_parameter_matched_global_lut"]),
        "global_p95": metrics["p95_error_ratio_to_parameter_matched_global_lut"]
        <= float(gates["maximum_p95_error_ratio_to_parameter_matched_global_lut"]),
        "global_worst": metrics["worst_error_ratio_to_parameter_matched_global_lut"]
        <= float(gates["maximum_worst_error_ratio_to_parameter_matched_global_lut"]),
        "bt0_mean": metrics["mean_improvement_over_closed_bt0_gain"]
        >= float(gates["minimum_mean_improvement_over_closed_bt0_gain"]),
        "bt0_wins": metrics["win_fraction_over_closed_bt0_gain"]
        >= float(gates["minimum_win_fraction_over_closed_bt0_gain"]),
        "luminance": metrics["mean_improvement_over_luminance_only"]
        >= float(gates["minimum_mean_improvement_over_luminance_only"]),
        "spatial": metrics["mean_improvement_over_spatial_only"]
        >= float(gates["minimum_mean_improvement_over_spatial_only"]),
        "style": metrics["median_style_retention"] >= float(gates["minimum_median_style_retention"]),
        "held_fit": metrics["p95_held_to_fit_error_ratio"]
        <= float(gates["maximum_p95_held_to_fit_error_ratio"]),
        "boundary": metrics["maximum_new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"]),
        "cube": metrics["maximum_out_of_cube_fraction"]
        <= float(gates["maximum_out_of_cube_fraction"]),
    }
    return {
        "schema": "neuro_film.u5_r2bt1_fivek_bilateral_affine_capacity_report.v1",
        "metrics": metrics,
        "gates": checks,
        "automatic_pass": all(checks.values()),
        "rows": output_rows,
    }


__all__ = [
    "FiveKBilateralAffineCapacityError",
    "_affine_design",
    "_apply_safe_affine",
    "evaluate_affine_capacity",
]
