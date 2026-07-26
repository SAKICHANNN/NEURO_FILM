"""Content-nuisance controls for bounded explicit palette-flow prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.roll2film.histogram_case_retrieval import (
    bound_score_grid,
    histogram_kde_score_grid,
    validate_histogram,
)


def content_residual_features(
    styled_histogram: np.ndarray, neutral_histogram: np.ndarray
) -> np.ndarray:
    styled = np.asarray(styled_histogram, dtype=np.float64)
    neutral = validate_histogram(neutral_histogram, styled.size)
    styled = validate_histogram(styled, styled.size)
    styled_root = np.sqrt(styled)
    neutral_root = np.sqrt(neutral)
    return np.concatenate(
        (styled_root, neutral_root, styled_root - neutral_root)
    )


def density_ratio_velocity_grid(
    styled_histogram: np.ndarray,
    neutral_histogram: np.ndarray,
    *,
    histogram_axis_size: int,
    bandwidth: float,
    velocity_grid_axis_size: int,
    score_difference_scale: float,
    coefficient_vector_norm_cap: float,
) -> np.ndarray:
    if not np.isfinite(score_difference_scale) or score_difference_scale <= 0.0:
        raise ValueError("score_difference_scale must be positive")
    styled_score = histogram_kde_score_grid(
        styled_histogram,
        histogram_axis_size=histogram_axis_size,
        bandwidth=bandwidth,
        velocity_grid_axis_size=velocity_grid_axis_size,
    )
    neutral_score = histogram_kde_score_grid(
        neutral_histogram,
        histogram_axis_size=histogram_axis_size,
        bandwidth=bandwidth,
        velocity_grid_axis_size=velocity_grid_axis_size,
    )
    return bound_score_grid(
        score_difference_scale * (styled_score - neutral_score),
        coefficient_vector_norm_cap=coefficient_vector_norm_cap,
    )


def project_velocity_vector_norms(
    grids: np.ndarray, *, maximum_norm: float
) -> np.ndarray:
    values = np.asarray(grids, dtype=np.float64)
    if (
        values.ndim != 5
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or not np.isfinite(maximum_norm)
        or maximum_norm <= 0.0
    ):
        raise ValueError("velocity grids or norm cap are invalid")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    scales = np.minimum(1.0, maximum_norm / np.maximum(norms, 1e-300))
    return values * scales


@dataclass(frozen=True)
class BoundedMultiOutputRidge:
    """Train-only-standardized ridge predictor for explicit velocity grids."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    grid_axis_size: int
    maximum_vector_norm: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        target_mean = np.asarray(self.target_mean, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        output_size = self.grid_axis_size**3 * 3
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or coefficients.shape != (len(mean), output_size)
            or target_mean.shape != (output_size,)
            or np.any(scale <= 0.0)
            or not all(
                np.all(np.isfinite(value))
                for value in (mean, scale, target_mean, coefficients)
            )
            or self.grid_axis_size < 2
            or self.maximum_vector_norm <= 0.0
        ):
            raise ValueError("ridge predictor state is invalid")
        for name, value in (
            ("feature_mean", mean),
            ("feature_scale", scale),
            ("target_mean", target_mean),
            ("coefficients", coefficients),
        ):
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.feature_mean)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("ridge features are invalid")
        normalized = (values - self.feature_mean) / self.feature_scale
        flat = normalized @ self.coefficients + self.target_mean
        grids = flat.reshape(
            len(values),
            self.grid_axis_size,
            self.grid_axis_size,
            self.grid_axis_size,
            3,
        )
        return project_velocity_vector_norms(
            grids, maximum_norm=self.maximum_vector_norm
        )


def fit_bounded_multi_output_ridge(
    features: np.ndarray,
    target_grids: np.ndarray,
    *,
    alpha: float,
    maximum_vector_norm: float,
) -> BoundedMultiOutputRidge:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(target_grids, dtype=np.float64)
    if (
        values.ndim != 2
        or targets.ndim != 5
        or len(values) != len(targets)
        or len(values) < 2
        or targets.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(targets))
        or not np.isfinite(alpha)
        or alpha <= 0.0
    ):
        raise ValueError("ridge training arrays or alpha are invalid")
    feature_mean = np.mean(values, axis=0)
    feature_scale = np.std(values, axis=0)
    feature_scale = np.maximum(feature_scale, 1e-8)
    normalized = (values - feature_mean) / feature_scale
    flat_targets = targets.reshape(len(targets), -1)
    target_mean = np.mean(flat_targets, axis=0)
    centered_targets = flat_targets - target_mean
    dual = np.linalg.solve(
        normalized @ normalized.T + alpha * np.eye(len(normalized)),
        centered_targets,
    )
    coefficients = normalized.T @ dual
    return BoundedMultiOutputRidge(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        coefficients=coefficients,
        grid_axis_size=targets.shape[1],
        maximum_vector_norm=maximum_vector_norm,
    )


__all__ = [
    "BoundedMultiOutputRidge",
    "content_residual_features",
    "density_ratio_velocity_grid",
    "fit_bounded_multi_output_ridge",
    "project_velocity_vector_norms",
]
