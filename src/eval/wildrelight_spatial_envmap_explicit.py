"""P302 explicit spatial environment-conditioned radiance operator primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class WildRelightSpatialOperatorError(ValueError):
    """Raised when P302 operator inputs violate the frozen contract."""


@dataclass(frozen=True)
class RidgeState:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray


def _rgb(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[2] != 3 or not np.isfinite(array).all():
        raise WildRelightSpatialOperatorError(f"{name} must be finite HxWx3 RGB")
    return array.astype(np.float64, copy=False)


def central_crop(values: np.ndarray, fraction: float) -> np.ndarray:
    array = _rgb(values, "values")
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise WildRelightSpatialOperatorError("crop fraction is invalid")
    height, width = array.shape[:2]
    crop_h = max(1, int(np.floor(height * fraction)))
    crop_w = max(1, int(np.floor(width * fraction)))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return array[y0 : y0 + crop_h, x0 : x0 + crop_w]


def log_grid_descriptor(
    values: np.ndarray, *, grid_rows: int, grid_columns: int, epsilon: float
) -> np.ndarray:
    array = _rgb(values, "source photo")
    if np.any(array < 0.0):
        raise WildRelightSpatialOperatorError("source photo contains negative radiance")
    if grid_rows <= 0 or grid_columns <= 0 or epsilon <= 0.0:
        raise WildRelightSpatialOperatorError("grid or epsilon is invalid")
    y_edges = np.linspace(0, array.shape[0], grid_rows + 1, dtype=np.int64)
    x_edges = np.linspace(0, array.shape[1], grid_columns + 1, dtype=np.int64)
    result = np.empty((grid_rows, grid_columns, 3), dtype=np.float64)
    for row in range(grid_rows):
        for column in range(grid_columns):
            cell = array[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            if cell.size == 0:
                raise WildRelightSpatialOperatorError("grid contains an empty cell")
            result[row, column] = np.median(np.log2(cell + epsilon), axis=(0, 1))
    return result


def source_dct_features(log_grid: np.ndarray) -> np.ndarray:
    grid = _rgb(log_grid, "source log grid")
    if grid.shape[0] != 4 or grid.shape[1] != 4:
        raise WildRelightSpatialOperatorError("source log grid must be 4x4x3")
    positions = np.arange(4, dtype=np.float64) + 0.5
    basis = np.empty((3, 4), dtype=np.float64)
    for frequency in range(3):
        scale = np.sqrt(1.0 / 4.0) if frequency == 0 else np.sqrt(2.0 / 4.0)
        basis[frequency] = scale * np.cos(np.pi * frequency * positions / 4.0)
    coefficients = np.einsum("ui,vj,ijc->uvc", basis, basis, grid, optimize=True)
    return coefficients.reshape(-1)


def spherical_harmonic_coefficients(envmap: np.ndarray, epsilon: float) -> np.ndarray:
    values = _rgb(envmap, "environment map")
    if np.any(values < 0.0) or epsilon <= 0.0:
        raise WildRelightSpatialOperatorError("environment map support is invalid")
    height, width = values.shape[:2]
    theta = np.pi * (np.arange(height, dtype=np.float64) + 0.5) / height
    phi = 2.0 * np.pi * (np.arange(width, dtype=np.float64) + 0.5) / width
    sin_theta = np.sin(theta)[:, None]
    x = sin_theta * np.cos(phi)[None, :]
    y = sin_theta * np.sin(phi)[None, :]
    z = np.cos(theta)[:, None] * np.ones((1, width), dtype=np.float64)
    basis = np.stack(
        (
            np.ones((height, width), dtype=np.float64),
            y,
            z,
            x,
            x * y,
            y * z,
            3.0 * z * z - 1.0,
            x * z,
            x * x - y * y,
        ),
        axis=2,
    )
    weights = np.broadcast_to(sin_theta, (height, width))
    normalizer = float(np.sum(weights))
    coefficients = (
        np.einsum("hw,hwk,hwc->kc", weights, basis, values, optimize=True) / normalizer
    )
    if np.any(coefficients[0] <= epsilon):
        raise WildRelightSpatialOperatorError(
            "environment degree-zero support is invalid"
        )
    return coefficients


def environment_difference_features(
    source_envmap: np.ndarray, target_envmap: np.ndarray, epsilon: float
) -> np.ndarray:
    source = spherical_harmonic_coefficients(source_envmap, epsilon)
    target = spherical_harmonic_coefficients(target_envmap, epsilon)
    normalized = (target - source) / np.maximum(np.abs(source[0:1]), epsilon)
    return normalized.reshape(-1)


def target_log_gain_grid(
    source: np.ndarray,
    target: np.ndarray,
    *,
    grid_rows: int,
    grid_columns: int,
    epsilon: float,
    minimum_log2_gain: float,
    maximum_log2_gain: float,
) -> np.ndarray:
    source_array = _rgb(source, "source photo")
    target_array = _rgb(target, "target photo")
    if source_array.shape != target_array.shape:
        raise WildRelightSpatialOperatorError("source and target shapes differ")
    if np.any(source_array < 0.0) or np.any(target_array < 0.0):
        raise WildRelightSpatialOperatorError("photo contains negative radiance")
    source_grid = log_grid_descriptor(
        source_array,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        epsilon=epsilon,
    )
    target_grid = log_grid_descriptor(
        target_array,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        epsilon=epsilon,
    )
    return np.clip(target_grid - source_grid, minimum_log2_gain, maximum_log2_gain)


def fit_ridge(
    features: np.ndarray, targets: np.ndarray, ridge_lambda: float
) -> RidgeState:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[0] != y.shape[0]
        or x.shape[0] < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(ridge_lambda)
        or ridge_lambda <= 0.0
    ):
        raise WildRelightSpatialOperatorError("ridge training arrays are invalid")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (x - mean) / scale
    design = np.concatenate(
        (np.ones((x.shape[0], 1), dtype=np.float64), standardized), axis=1
    )
    penalty = ridge_lambda * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RidgeState(mean.copy(), scale.copy(), coefficients.copy())


def predict_ridge(state: RidgeState, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != state.feature_mean.size or not np.isfinite(x).all():
        raise WildRelightSpatialOperatorError("ridge prediction features are invalid")
    standardized = (x - state.feature_mean) / state.feature_scale
    design = np.concatenate(
        (np.ones((x.shape[0], 1), dtype=np.float64), standardized), axis=1
    )
    return design @ state.coefficients


def apply_log_gain_grid(
    source: np.ndarray,
    log_gain_grid: np.ndarray,
    *,
    minimum_log2_gain: float,
    maximum_log2_gain: float,
) -> np.ndarray:
    values = _rgb(source, "source photo")
    grid = _rgb(log_gain_grid, "log gain grid")
    if np.any(values < 0.0) or grid.shape != (4, 4, 3):
        raise WildRelightSpatialOperatorError("source or gain grid is invalid")
    grid = np.clip(grid, minimum_log2_gain, maximum_log2_gain)
    y_grid = (np.arange(4, dtype=np.float64) + 0.5) * values.shape[0] / 4.0 - 0.5
    x_grid = (np.arange(4, dtype=np.float64) + 0.5) * values.shape[1] / 4.0 - 0.5
    y_pixels = np.arange(values.shape[0], dtype=np.float64)
    x_pixels = np.arange(values.shape[1], dtype=np.float64)
    gain = np.empty_like(values, dtype=np.float64)
    for channel in range(3):
        horizontal = np.stack(
            [np.interp(x_pixels, x_grid, row) for row in grid[:, :, channel]],
            axis=0,
        )
        for column in range(values.shape[1]):
            gain[:, column, channel] = np.interp(
                y_pixels, y_grid, horizontal[:, column]
            )
    output = values * np.exp2(gain)
    if not np.isfinite(output).all():
        raise WildRelightSpatialOperatorError(
            "explicit operator produced nonfinite output"
        )
    return output.astype(np.float32)


__all__ = [
    "RidgeState",
    "WildRelightSpatialOperatorError",
    "apply_log_gain_grid",
    "central_crop",
    "environment_difference_features",
    "fit_ridge",
    "log_grid_descriptor",
    "predict_ridge",
    "source_dct_features",
    "spherical_harmonic_coefficients",
    "target_log_gain_grid",
]
