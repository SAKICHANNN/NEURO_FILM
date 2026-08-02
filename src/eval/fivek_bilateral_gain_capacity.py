"""Held-block capacity test for a bounded spatial/luminance gain field."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.fivek_adaptive_lut_basis_development import _trilinear_features


class FiveKBilateralGainCapacityError(ValueError):
    """Raised when the frozen capacity contract is violated."""


def _rgb(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[-1] != 3
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or np.any(array > 1.0)
    ):
        raise FiveKBilateralGainCapacityError("invalid bounded encoded RGB")
    return array


def _stable_midrank(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranked = np.empty(len(flat), dtype=np.float64)
    start = 0
    while start < len(flat):
        end = start + 1
        while end < len(flat) and flat[order[end]] == flat[order[start]]:
            end += 1
        ranked[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    if len(flat) > 1:
        ranked /= len(flat) - 1
    else:
        ranked.fill(0.5)
    return ranked.reshape(values.shape)


def _axis_weights(values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    if size < 1:
        raise FiveKBilateralGainCapacityError("grid axis must be positive")
    if size == 1:
        return np.zeros(values.shape, dtype=np.int64), np.zeros(values.shape)
    scaled = np.clip(values, 0.0, 1.0) * (size - 1)
    lower = np.minimum(np.floor(scaled).astype(np.int64), size - 2)
    fraction = scaled - lower
    return lower, fraction


def _gain_features(
    shape: tuple[int, int], luma_rank: np.ndarray, grid: Sequence[int]
) -> np.ndarray:
    rows, columns = shape
    gy, gx, gl = map(int, grid)
    if luma_rank.shape != shape:
        raise FiveKBilateralGainCapacityError("luminance-rank shape mismatch")
    y = np.linspace(0.0, 1.0, rows, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, columns, dtype=np.float64)[None, :]
    y = np.broadcast_to(y, shape)
    x = np.broadcast_to(x, shape)
    iy, fy = _axis_weights(y, gy)
    ix, fx = _axis_weights(x, gx)
    il, fl = _axis_weights(luma_rank, gl)
    features = np.zeros((rows * columns, gy * gx * gl), dtype=np.float64)
    flat_index = np.arange(rows * columns)
    for dy in range(1 if gy == 1 else 2):
        wy = np.ones(shape) if gy == 1 else (fy if dy else 1.0 - fy)
        jy = iy if gy == 1 else iy + dy
        for dx in range(1 if gx == 1 else 2):
            wx = np.ones(shape) if gx == 1 else (fx if dx else 1.0 - fx)
            jx = ix if gx == 1 else ix + dx
            for dl in range(1 if gl == 1 else 2):
                wl = np.ones(shape) if gl == 1 else (fl if dl else 1.0 - fl)
                jl = il if gl == 1 else il + dl
                node = ((jy * gx + jx) * gl + jl).reshape(-1)
                features[flat_index, node] += (wy * wx * wl).reshape(-1)
    if not np.allclose(features.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise FiveKBilateralGainCapacityError("gain interpolation is not convex")
    return features


def _difference_matrix(grid: Sequence[int]) -> np.ndarray:
    gy, gx, gl = map(int, grid)
    rows: list[np.ndarray] = []
    for y in range(gy):
        for x in range(gx):
            for luma in range(gl):
                here = (y * gx + x) * gl + luma
                for axis, limit in enumerate((gy, gx, gl)):
                    coordinate = (y, x, luma)[axis]
                    if coordinate + 1 >= limit:
                        continue
                    peer = [y, x, luma]
                    peer[axis] += 1
                    there = (peer[0] * gx + peer[1]) * gl + peer[2]
                    row = np.zeros(gy * gx * gl, dtype=np.float64)
                    row[here] = 1.0
                    row[there] = -1.0
                    rows.append(row)
    return np.stack(rows) if rows else np.zeros((0, gy * gx * gl))


def _fit_gain(
    features: np.ndarray,
    source_linear: np.ndarray,
    target_linear: np.ndarray,
    fit_mask: np.ndarray,
    spec: Mapping[str, Any],
) -> np.ndarray:
    epsilon = float(spec["epsilon"])
    bound = float(spec["log_gain_bound"])
    target = np.log((target_linear + epsilon) / (source_linear + epsilon))
    target = np.clip(target, -bound, bound).reshape(-1, 3)
    selected = fit_mask.reshape(-1)
    design = features[selected]
    differences = _difference_matrix(spec["grid_shape"])
    normal = design.T @ design
    normal += float(spec["ridge"]) * np.eye(normal.shape[0])
    normal += float(spec["first_difference_smoothness"]) * (differences.T @ differences)
    coefficients = np.linalg.solve(normal, design.T @ target[selected])
    return np.clip(coefficients, -bound, bound)


def _apply_safe_log_gain(
    source_linear: np.ndarray,
    features: np.ndarray,
    coefficients: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    epsilon = float(spec["epsilon"])
    bound = float(spec["log_gain_bound"])
    source = np.asarray(source_linear, dtype=np.float64).reshape(-1, 3)
    gain = np.clip(features @ coefficients, -bound, bound)
    dose = np.ones(len(source), dtype=np.float64)
    boundary = np.any((source <= epsilon) | (source >= 1.0 - epsilon), axis=1)
    dose[boundary] = 0.0
    positive = gain > 0.0
    negative = gain < 0.0
    upper = np.full_like(gain, np.inf)
    lower = np.full_like(gain, np.inf)
    upper[positive] = np.log((1.0 - epsilon) / source[positive]) / gain[positive]
    lower[negative] = np.log(epsilon / source[negative]) / gain[negative]
    dose = np.minimum(dose, np.min(np.minimum(upper, lower), axis=1))
    dose = np.clip(dose, 0.0, 1.0)
    output = source * np.exp(dose[:, None] * gain)
    if (
        not np.all(np.isfinite(output))
        or np.any(output < -1.0e-12)
        or np.any(output > 1.0 + 1.0e-12)
    ):
        raise FiveKBilateralGainCapacityError("analytical gain guard failed")
    return output.reshape(source_linear.shape), dose.reshape(source_linear.shape[:2])


def _fit_global_lut(
    source: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    spec: Mapping[str, Any],
    epsilon: float,
) -> np.ndarray:
    size = int(spec["grid_shape"][0])
    if list(spec["grid_shape"]) != [size, size, size]:
        raise FiveKBilateralGainCapacityError("global LUT must be cubic")
    features = _trilinear_features(source, size).reshape(-1, size**3)
    headroom = 0.995 * np.maximum(
        0.0, np.minimum(source - epsilon, 1.0 - epsilon - source)
    ).reshape(-1, 3)
    delta = (target - source).reshape(-1, 3)
    selected = fit_mask.reshape(-1)
    differences = _difference_matrix((size, size, size))
    penalty = float(spec["ridge"]) * np.eye(size**3)
    penalty += float(spec["first_difference_smoothness"]) * (
        differences.T @ differences
    )
    coefficients = np.empty((size**3, 3), dtype=np.float64)
    for channel in range(3):
        design = features[selected] * headroom[selected, channel, None]
        normal = design.T @ design + penalty
        coefficients[:, channel] = np.linalg.solve(
            normal, design.T @ delta[selected, channel]
        )
    return np.clip(coefficients, -1.0, 1.0)


def _apply_global_lut(
    source: np.ndarray, coefficients: np.ndarray, epsilon: float
) -> np.ndarray:
    size = round(coefficients.shape[0] ** (1.0 / 3.0))
    features = _trilinear_features(source, size).reshape(-1, size**3)
    residual = (features @ coefficients).reshape(source.shape)
    headroom = 0.995 * np.maximum(
        0.0, np.minimum(source - epsilon, 1.0 - epsilon - source)
    )
    output = source + headroom * residual
    if np.any(output < 0.0) or np.any(output > 1.0):
        raise FiveKBilateralGainCapacityError("global LUT escaped cube")
    return output


def _held_mask(
    shape: tuple[int, int], spec: Mapping[str, Any], fold: int
) -> np.ndarray:
    rows, columns = shape
    macro_rows = int(spec["macro_rows"])
    macro_columns = int(spec["macro_columns"])
    yy = np.minimum(np.arange(rows) * macro_rows // rows, macro_rows - 1)
    xx = np.minimum(np.arange(columns) * macro_columns // columns, macro_columns - 1)
    return ((yy[:, None] + xx[None, :]) % 2) == fold


def _rmse(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    delta = np.asarray(first)[mask] - np.asarray(second)[mask]
    return float(np.sqrt(np.mean(delta * delta)))


def _new_boundary_fraction(
    source: np.ndarray, output: np.ndarray, mask: np.ndarray, epsilon: float
) -> float:
    source_boundary = np.any((source <= epsilon) | (source >= 1.0 - epsilon), axis=-1)
    output_boundary = np.any((output <= epsilon) | (output >= 1.0 - epsilon), axis=-1)
    return float(np.mean((output_boundary & ~source_boundary)[mask]))


def evaluate_gain_capacity(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    expected = int(config["population"]["source_count_exact"])
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    if (
        len(ordered) != expected
        or len({str(row["group"]) for row in ordered}) != expected
    ):
        raise FiveKBilateralGainCapacityError("population identity drift")
    candidate = config["candidate"]
    held_spec = config["held_block_protocol"]
    controls = config["controls"]
    boundary_epsilon = float(config["evaluation"]["boundary_epsilon"])
    output_rows: list[dict[str, Any]] = []
    for row in ordered:
        source_encoded = _rgb(row["source"])
        target_encoded = _rgb(row["target"])
        if source_encoded.shape != target_encoded.shape:
            raise FiveKBilateralGainCapacityError("paired shape mismatch")
        source_linear = encoded_srgb_to_linear(source_encoded)
        target_linear = encoded_srgb_to_linear(target_encoded)
        luma = np.tensordot(
            source_linear, np.asarray([0.2126, 0.7152, 0.0722]), axes=([-1], [0])
        )
        rank = _stable_midrank(luma)
        candidate_features = _gain_features(
            source_linear.shape[:2], rank, candidate["grid_shape"]
        )
        luma_spec = {**candidate, **controls["luminance_only_gain"]}
        spatial_spec = {**candidate, **controls["spatial_only_gain"]}
        luma_features = _gain_features(
            source_linear.shape[:2], rank, luma_spec["grid_shape"]
        )
        spatial_features = _gain_features(
            source_linear.shape[:2], rank, spatial_spec["grid_shape"]
        )
        for fold in map(int, held_spec["folds"]):
            held = _held_mask(source_linear.shape[:2], held_spec, fold)
            fit = ~held
            candidate_coefficients = _fit_gain(
                candidate_features, source_linear, target_linear, fit, candidate
            )
            luma_coefficients = _fit_gain(
                luma_features, source_linear, target_linear, fit, luma_spec
            )
            spatial_coefficients = _fit_gain(
                spatial_features, source_linear, target_linear, fit, spatial_spec
            )
            lut_coefficients = _fit_global_lut(
                source_encoded,
                target_encoded,
                fit,
                controls["parameter_matched_global_rgb_lut"],
                boundary_epsilon,
            )
            candidate_linear, dose = _apply_safe_log_gain(
                source_linear, candidate_features, candidate_coefficients, candidate
            )
            luma_linear, _ = _apply_safe_log_gain(
                source_linear, luma_features, luma_coefficients, luma_spec
            )
            spatial_linear, _ = _apply_safe_log_gain(
                source_linear, spatial_features, spatial_coefficients, spatial_spec
            )
            candidate_encoded = linear_srgb_to_encoded(candidate_linear)
            luma_encoded = linear_srgb_to_encoded(luma_linear)
            spatial_encoded = linear_srgb_to_encoded(spatial_linear)
            global_encoded = _apply_global_lut(
                source_encoded, lut_coefficients, boundary_epsilon
            )
            held_identity = _rmse(source_encoded, target_encoded, held)
            output_rows.append(
                {
                    "pair_id": str(row["pair_id"]),
                    "group": str(row["group"]),
                    "target_variant": str(row["target_variant"]),
                    "fold": fold,
                    "fit_pixels": int(np.sum(fit)),
                    "held_pixels": int(np.sum(held)),
                    "identity_rmse": held_identity,
                    "candidate_fit_rmse": _rmse(candidate_encoded, target_encoded, fit),
                    "candidate_rmse": _rmse(candidate_encoded, target_encoded, held),
                    "global_lut_rmse": _rmse(global_encoded, target_encoded, held),
                    "luminance_only_rmse": _rmse(luma_encoded, target_encoded, held),
                    "spatial_only_rmse": _rmse(spatial_encoded, target_encoded, held),
                    "candidate_style_retention": _rmse(
                        candidate_encoded, source_encoded, held
                    )
                    / max(held_identity, 1.0e-12),
                    "minimum_gain_dose": float(np.min(dose[held])),
                    "new_boundary_fraction": _new_boundary_fraction(
                        source_encoded,
                        candidate_encoded,
                        held,
                        boundary_epsilon,
                    ),
                    "out_of_cube_fraction": float(
                        np.mean((candidate_encoded < 0.0) | (candidate_encoded > 1.0))
                    ),
                }
            )
    candidate_error = np.asarray([row["candidate_rmse"] for row in output_rows])
    global_error = np.asarray([row["global_lut_rmse"] for row in output_rows])
    luma_error = np.asarray([row["luminance_only_rmse"] for row in output_rows])
    spatial_error = np.asarray([row["spatial_only_rmse"] for row in output_rows])
    fit_error = np.asarray([row["candidate_fit_rmse"] for row in output_rows])
    metrics = {
        "rows": len(output_rows),
        "mean_improvement_over_parameter_matched_global_lut": float(
            (global_error.mean() - candidate_error.mean()) / global_error.mean()
        ),
        "win_fraction_over_parameter_matched_global_lut": float(
            np.mean(candidate_error < global_error)
        ),
        "p95_error_ratio_to_parameter_matched_global_lut": float(
            np.quantile(candidate_error, 0.95) / np.quantile(global_error, 0.95)
        ),
        "worst_error_ratio_to_parameter_matched_global_lut": float(
            np.max(candidate_error) / np.max(global_error)
        ),
        "mean_improvement_over_luminance_only": float(
            (luma_error.mean() - candidate_error.mean()) / luma_error.mean()
        ),
        "mean_improvement_over_spatial_only": float(
            (spatial_error.mean() - candidate_error.mean()) / spatial_error.mean()
        ),
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
        "minimum_gain_dose": float(
            min(row["minimum_gain_dose"] for row in output_rows)
        ),
        "candidate_error_sha256": hashlib.sha256(
            np.ascontiguousarray(candidate_error, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    gates = config["evaluation"]["automatic_gates"]
    checks = {
        "source_count": len(ordered) == int(gates["source_count_exact"]),
        "target_variants": len({row["target_variant"] for row in output_rows})
        == int(gates["target_variant_count_exact"]),
        "folds": len({row["fold"] for row in output_rows})
        == int(gates["fold_count_exact"]),
        "global_mean": metrics["mean_improvement_over_parameter_matched_global_lut"]
        >= float(gates["minimum_mean_improvement_over_parameter_matched_global_lut"]),
        "global_wins": metrics["win_fraction_over_parameter_matched_global_lut"]
        >= float(gates["minimum_win_fraction_over_parameter_matched_global_lut"]),
        "global_p95": metrics["p95_error_ratio_to_parameter_matched_global_lut"]
        <= float(gates["maximum_p95_error_ratio_to_parameter_matched_global_lut"]),
        "global_worst": metrics["worst_error_ratio_to_parameter_matched_global_lut"]
        <= float(gates["maximum_worst_error_ratio_to_parameter_matched_global_lut"]),
        "luminance": metrics["mean_improvement_over_luminance_only"]
        >= float(gates["minimum_mean_improvement_over_luminance_only"]),
        "spatial": metrics["mean_improvement_over_spatial_only"]
        >= float(gates["minimum_mean_improvement_over_spatial_only"]),
        "style": metrics["median_style_retention"]
        >= float(gates["minimum_median_style_retention"]),
        "held_fit": metrics["p95_held_to_fit_error_ratio"]
        <= float(gates["maximum_p95_held_to_fit_error_ratio"]),
        "boundary": metrics["maximum_new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"]),
        "cube": metrics["maximum_out_of_cube_fraction"]
        <= float(gates["maximum_out_of_cube_fraction"]),
    }
    return {
        "schema": "neuro_film.u5_r2bt0_fivek_bilateral_gain_capacity_report.v1",
        "metrics": metrics,
        "gates": checks,
        "automatic_pass": all(checks.values()),
        "rows": output_rows,
    }


__all__ = [
    "FiveKBilateralGainCapacityError",
    "_apply_safe_log_gain",
    "_gain_features",
    "_held_mask",
    "_stable_midrank",
    "evaluate_gain_capacity",
]
