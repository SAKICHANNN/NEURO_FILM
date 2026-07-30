"""Optical-density factorization of one explicit colour-operator residual."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .factorized_boundary_guard import (
    _maximum_safe_scale,
    _source_inclusive_rails,
)
from .positive_film import PositiveFilmResponseOperator


@dataclass(frozen=True)
class DensityResidualGuardResult:
    output: np.ndarray
    neutral_scale: np.ndarray
    opponent_scale: np.ndarray


def _density_coordinate(rgb: np.ndarray, floor: float) -> np.ndarray:
    return -np.log10((rgb + floor) / (1.0 + floor))


def _density_inverse(density: np.ndarray, floor: float) -> np.ndarray:
    return (1.0 + floor) * np.power(10.0, -density) - floor


def apply_target_density_residual_guard(
    linear_rgb: np.ndarray,
    target_linear_rgb: np.ndarray,
    *,
    neutral_strength: float,
    opponent_strength: float,
    neutral_weights: np.ndarray,
    density_floor: float,
    hard_boundary_epsilon_encoded_srgb: float,
    guard_boundary_epsilon_encoded_srgb: float,
) -> DensityResidualGuardResult:
    """Apply a target residual as neutral plus opponent optical density.

    The soft-floor density coordinate maps RGB zero and one exactly.  A scalar
    neutral-density delta is applied equally to all channels, followed by the
    remaining opponent-density delta.  Each stage uses one analytical shared
    scale per pixel, so it cannot rotate the requested density direction or
    rely on output clipping.
    """

    source = np.asarray(linear_rgb, dtype=np.float64)
    target = np.asarray(target_linear_rgb, dtype=np.float64)
    weights = np.asarray(neutral_weights, dtype=np.float64)
    if (
        source.ndim < 2
        or source.shape[-1] != 3
        or target.shape != source.shape
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or np.any(target < 0.0)
        or np.any(target > 1.0)
        or weights.shape != (3,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or abs(float(np.sum(weights)) - 1.0) > 1e-12
        or not np.isfinite(density_floor)
        or density_floor <= 0.0
        or density_floor > 2.0**-8
        or not np.isfinite(neutral_strength)
        or neutral_strength < 0.0
        or neutral_strength > 1.0
        or not np.isfinite(opponent_strength)
        or opponent_strength < 0.0
        or opponent_strength > 2.0
    ):
        raise ValueError("invalid density residual guard inputs")
    if np.array_equal(source, target):
        ones = np.ones(source.shape[:-1], dtype=np.float64)
        return DensityResidualGuardResult(
            output=source.copy(),
            neutral_scale=ones,
            opponent_scale=ones.copy(),
        )

    lower_rgb, upper_rgb = _source_inclusive_rails(
        source,
        hard_boundary_epsilon_encoded_srgb=(
            hard_boundary_epsilon_encoded_srgb
        ),
        guard_boundary_epsilon_encoded_srgb=(
            guard_boundary_epsilon_encoded_srgb
        ),
    )
    density = _density_coordinate(source, density_floor)
    target_density = _density_coordinate(target, density_floor)
    lower_density = _density_coordinate(upper_rgb, density_floor)
    upper_density = _density_coordinate(lower_rgb, density_floor)

    residual = target_density - density
    neutral_delta = neutral_strength * (residual @ weights)[..., None]
    neutral_scale = _maximum_safe_scale(
        density,
        neutral_delta,
        lower_density,
        upper_density,
    )
    neutral_output = density + neutral_scale[..., None] * neutral_delta

    opponent_delta = opponent_strength * (
        residual - (residual @ weights)[..., None]
    )
    opponent_scale = _maximum_safe_scale(
        neutral_output,
        opponent_delta,
        lower_density,
        upper_density,
    )
    output_density = (
        neutral_output + opponent_scale[..., None] * opponent_delta
    )
    output = _density_inverse(output_density, density_floor)
    tolerance = 32.0 * np.finfo(np.float64).eps
    if (
        not np.all(np.isfinite(output))
        or np.any(output < lower_rgb - tolerance)
        or np.any(output > upper_rgb + tolerance)
    ):
        raise RuntimeError("density residual guard escaped source-inclusive rails")
    return DensityResidualGuardResult(
        output=output,
        neutral_scale=neutral_scale,
        opponent_scale=opponent_scale,
    )


def apply_density_residual_guard(
    operator: PositiveFilmResponseOperator,
    linear_rgb: np.ndarray,
    **kwargs: object,
) -> DensityResidualGuardResult:
    """Apply the density factorization to one explicit positive-film operator."""

    source = np.asarray(linear_rgb, dtype=np.float64)
    return apply_target_density_residual_guard(
        source,
        operator.apply(source),
        **kwargs,
    )


__all__ = [
    "DensityResidualGuardResult",
    "apply_density_residual_guard",
    "apply_target_density_residual_guard",
]
