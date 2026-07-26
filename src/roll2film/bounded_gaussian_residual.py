"""Bounded fixed-geometry Gaussian residual colour operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BOUNDED_GAUSSIAN_RESIDUAL_SCHEMA = "roll2film.bounded_gaussian_residual.v1"


def regular_rgb_centers(axis_size: int) -> np.ndarray:
    """Return a deterministic regular grid of RGB Gaussian centres."""

    if axis_size < 2:
        raise ValueError("axis_size must be at least two")
    axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim < 2
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("rgb must be finite [0, 1] data with shape (..., 3)")
    return values


def normalized_gaussian_weights(
    rgb: np.ndarray,
    centers: np.ndarray,
    *,
    sigma: float,
    epsilon: float,
) -> np.ndarray:
    """Evaluate normalized equal-covariance Gaussian weights."""

    values = _validate_rgb(rgb)
    center_values = np.asarray(centers, dtype=np.float64)
    if (
        center_values.ndim != 2
        or center_values.shape[1] != 3
        or len(center_values) == 0
        or not np.all(np.isfinite(center_values))
        or np.any(center_values < 0.0)
        or np.any(center_values > 1.0)
    ):
        raise ValueError("centers must be finite [0, 1] rows with shape (N, 3)")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    flat = values.reshape(-1, 3)
    squared = np.sum((flat[:, None, :] - center_values[None, :, :]) ** 2, axis=2)
    normalization = ((2.0 * np.pi) ** 1.5) * (sigma**3)
    density = np.exp(-0.5 * squared / (sigma * sigma)) / normalization
    weights = density / (np.sum(density, axis=1, keepdims=True) + epsilon)
    return weights.reshape(values.shape[:-1] + (len(center_values),))


def gaussian_affine_design(
    rgb: np.ndarray,
    centers: np.ndarray,
    *,
    sigma: float,
    epsilon: float,
) -> np.ndarray:
    """Build global and Gaussian-weighted local affine residual features."""

    values = _validate_rgb(rgb)
    flat = values.reshape(-1, 3)
    affine = np.column_stack((np.ones(len(flat), dtype=np.float64), flat))
    weights = normalized_gaussian_weights(
        flat,
        centers,
        sigma=sigma,
        epsilon=epsilon,
    )
    local = (weights[:, :, None] * affine[:, None, :]).reshape(len(flat), -1)
    return np.column_stack((affine, local))


def signed_headroom_map(rgb: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply an analytically range-bounded signed residual."""

    values = _validate_rgb(rgb)
    residual = np.asarray(delta, dtype=np.float64)
    if residual.shape != values.shape or not np.all(np.isfinite(residual)):
        raise ValueError("delta must be finite and have the same shape as rgb")
    positive = np.maximum(residual, 0.0)
    negative = np.minimum(residual, 0.0)
    return (
        values
        + (1.0 - values) * np.tanh(positive)
        + values * np.tanh(negative)
    )


def inverse_signed_headroom(rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the finite residual whose bounded map approximates ``target``."""

    values = _validate_rgb(rgb)
    targets = _validate_rgb(target)
    if targets.shape != values.shape:
        raise ValueError("target must have the same shape as rgb")
    rising = targets >= values
    positive_scale = np.maximum(1.0 - values, np.finfo(np.float64).eps)
    negative_scale = np.maximum(values, np.finfo(np.float64).eps)
    ratio = np.where(
        rising,
        (targets - values) / positive_scale,
        (targets - values) / negative_scale,
    )
    ratio = np.clip(ratio, -1.0 + 1e-9, 1.0 - 1e-9)
    return np.arctanh(ratio)


@dataclass(frozen=True)
class BoundedGaussianResidualOperator:
    """Fixed Gaussian geometry with fitted local affine residual functions."""

    centers: np.ndarray
    sigma: float
    coefficients: np.ndarray
    epsilon: float = 1e-12
    working_space: str = "linear_srgb_d65"

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if (
            centers.ndim != 2
            or centers.shape[1] != 3
            or len(centers) == 0
            or not np.all(np.isfinite(centers))
            or np.any(centers < 0.0)
            or np.any(centers > 1.0)
        ):
            raise ValueError("centers must be finite [0, 1] rows")
        expected_features = 4 * (len(centers) + 1)
        if (
            coefficients.shape != (expected_features, 3)
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError(
                f"coefficients must have shape ({expected_features}, 3)"
            )
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.working_space != "linear_srgb_d65":
            raise ValueError("v1 requires linear_srgb_d65")
        centers = centers.copy()
        coefficients = coefficients.copy()
        centers.setflags(write=False)
        coefficients.setflags(write=False)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "coefficients", coefficients)

    @classmethod
    def identity(
        cls,
        *,
        axis_size: int,
        sigma: float,
        epsilon: float = 1e-12,
    ) -> "BoundedGaussianResidualOperator":
        centers = regular_rgb_centers(axis_size)
        return cls(
            centers=centers,
            sigma=sigma,
            epsilon=epsilon,
            coefficients=np.zeros((4 * (len(centers) + 1), 3), dtype=np.float64),
        )

    def residual(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        design = gaussian_affine_design(
            values,
            self.centers,
            sigma=self.sigma,
            epsilon=self.epsilon,
        )
        return (design @ self.coefficients).reshape(values.shape)

    def apply(self, rgb: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        values = _validate_rgb(rgb)
        if not np.isfinite(strength) or strength < 0.0 or strength > 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        if strength == 0.0:
            return values.copy()
        return signed_headroom_map(values, strength * self.residual(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BOUNDED_GAUSSIAN_RESIDUAL_SCHEMA,
            "working_space": self.working_space,
            "sigma": self.sigma,
            "epsilon": self.epsilon,
            "centers": self.centers.tolist(),
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoundedGaussianResidualOperator":
        if payload.get("schema") != BOUNDED_GAUSSIAN_RESIDUAL_SCHEMA:
            raise ValueError("unsupported bounded Gaussian residual schema")
        return cls(
            centers=np.asarray(payload["centers"], dtype=np.float64),
            sigma=float(payload["sigma"]),
            epsilon=float(payload["epsilon"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            working_space=str(payload["working_space"]),
        )


def fit_bounded_gaussian_residual(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    axis_size: int,
    sigma: float,
    epsilon: float,
    ridge: float,
) -> BoundedGaussianResidualOperator:
    """Fit residual coefficients deterministically with ridge regression."""

    values = _validate_rgb(rgb)
    targets = _validate_rgb(target)
    if targets.shape != values.shape:
        raise ValueError("target must have the same shape as rgb")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    centers = regular_rgb_centers(axis_size)
    design = gaussian_affine_design(
        values,
        centers,
        sigma=sigma,
        epsilon=epsilon,
    )
    desired = inverse_signed_headroom(values, targets).reshape(-1, 3)
    gram = design.T @ design
    regularizer = ridge * np.eye(gram.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(gram + regularizer, design.T @ desired)
    return BoundedGaussianResidualOperator(
        centers=centers,
        sigma=sigma,
        epsilon=epsilon,
        coefficients=coefficients,
    )


def finite_difference_jacobians(
    operator: BoundedGaussianResidualOperator,
    points: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    values = _validate_rgb(points)
    if (
        values.ndim != 2
        or not np.isfinite(step)
        or step <= 0.0
        or np.any(values < step)
        or np.any(values > 1.0 - step)
    ):
        raise ValueError("points must be finite interior RGB rows for the given step")
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        columns.append(
            (operator.apply(values + offset) - operator.apply(values - offset))
            / (2.0 * step)
        )
    return np.stack(columns, axis=-1)


def published_formula_initialization_witness(
    rgb: np.ndarray,
    *,
    axis_size: int,
    sigma: float,
    epsilon: float,
    identity_global: bool,
) -> np.ndarray:
    """Evaluate the paper equation with its stated local identity initialization."""

    values = _validate_rgb(rgb)
    centers = regular_rgb_centers(axis_size)
    weights = normalized_gaussian_weights(
        values,
        centers,
        sigma=sigma,
        epsilon=epsilon,
    )
    local = np.sum(weights[..., None] * values[..., None, :], axis=-2)
    return local + (values if identity_global else 0.0)


__all__ = [
    "BOUNDED_GAUSSIAN_RESIDUAL_SCHEMA",
    "BoundedGaussianResidualOperator",
    "finite_difference_jacobians",
    "fit_bounded_gaussian_residual",
    "gaussian_affine_design",
    "inverse_signed_headroom",
    "normalized_gaussian_weights",
    "published_formula_initialization_witness",
    "regular_rgb_centers",
    "signed_headroom_map",
]
