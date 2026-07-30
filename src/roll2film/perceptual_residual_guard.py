"""CIELAB lightness/chroma factorization with analytical RGB gamut guards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab

from .factorized_boundary_guard import (
    _maximum_safe_scale,
    _source_inclusive_rails,
)
from .positive_film import PositiveFilmResponseOperator


@dataclass(frozen=True)
class PerceptualResidualGuardResult:
    output: np.ndarray
    lightness_scale: np.ndarray
    chroma_scale: np.ndarray


def _as_lab(rgb: np.ndarray) -> np.ndarray:
    return linear_rgb_to_lab(
        np.ascontiguousarray(rgb, dtype=np.float32),
        working_space="linear_srgb",
    ).astype(np.float64)


def _as_linear_rgb(lab: np.ndarray) -> np.ndarray:
    return lab_to_linear_rgb(
        np.ascontiguousarray(lab, dtype=np.float32),
        working_space="linear_srgb",
    ).astype(np.float64)


def apply_target_perceptual_residual_guard(
    linear_rgb: np.ndarray,
    target_linear_rgb: np.ndarray,
    *,
    lightness_strength: float,
    chroma_strength: float,
    hard_boundary_epsilon_encoded_srgb: float,
    guard_boundary_epsilon_encoded_srgb: float,
) -> PerceptualResidualGuardResult:
    """Apply target CIELAB L* and a*/b* residuals without output clipping.

    The two requested perceptual stages are converted back to linear sRGB and
    analytically shortened along their actual RGB direction only when required
    by source-inclusive rails.  No per-image fit, hue rotation, or hard clip is
    performed.
    """

    source = np.asarray(linear_rgb, dtype=np.float64)
    target = np.asarray(target_linear_rgb, dtype=np.float64)
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
        or not np.isfinite(lightness_strength)
        or lightness_strength < 0.0
        or lightness_strength > 1.0
        or not np.isfinite(chroma_strength)
        or chroma_strength < 0.0
        or chroma_strength > 2.0
    ):
        raise ValueError("invalid perceptual residual guard inputs")
    if np.array_equal(source, target):
        ones = np.ones(source.shape[:-1], dtype=np.float64)
        return PerceptualResidualGuardResult(
            output=source.copy(),
            lightness_scale=ones,
            chroma_scale=ones.copy(),
        )

    lower, upper = _source_inclusive_rails(
        source,
        hard_boundary_epsilon_encoded_srgb=(
            hard_boundary_epsilon_encoded_srgb
        ),
        guard_boundary_epsilon_encoded_srgb=(
            guard_boundary_epsilon_encoded_srgb
        ),
    )
    source_lab = _as_lab(source)
    target_lab = _as_lab(target)
    residual_lab = target_lab - source_lab

    requested_lightness_lab = source_lab.copy()
    requested_lightness_lab[..., 0] += (
        lightness_strength * residual_lab[..., 0]
    )
    requested_lightness_rgb = _as_linear_rgb(requested_lightness_lab)
    lightness_delta = requested_lightness_rgb - source
    lightness_scale = _maximum_safe_scale(
        source, lightness_delta, lower, upper
    )
    lightness_output = (
        source + lightness_scale[..., None] * lightness_delta
    )

    lightness_output_lab = _as_lab(lightness_output)
    requested_chroma_lab = lightness_output_lab.copy()
    requested_chroma_lab[..., 1:3] += (
        chroma_strength * residual_lab[..., 1:3]
    )
    requested_chroma_rgb = _as_linear_rgb(requested_chroma_lab)
    chroma_delta = requested_chroma_rgb - lightness_output
    chroma_scale = _maximum_safe_scale(
        lightness_output, chroma_delta, lower, upper
    )
    output = lightness_output + chroma_scale[..., None] * chroma_delta

    tolerance = 64.0 * np.finfo(np.float64).eps
    if (
        not np.all(np.isfinite(output))
        or np.any(output < lower - tolerance)
        or np.any(output > upper + tolerance)
    ):
        raise RuntimeError(
            "perceptual residual guard escaped source-inclusive rails"
        )
    return PerceptualResidualGuardResult(
        output=output,
        lightness_scale=lightness_scale,
        chroma_scale=chroma_scale,
    )


def apply_perceptual_residual_guard(
    operator: PositiveFilmResponseOperator,
    linear_rgb: np.ndarray,
    **kwargs: object,
) -> PerceptualResidualGuardResult:
    """Apply the perceptual factorization to one explicit film operator."""

    source = np.asarray(linear_rgb, dtype=np.float64)
    return apply_target_perceptual_residual_guard(
        source,
        operator.apply(source),
        **kwargs,
    )


__all__ = [
    "PerceptualResidualGuardResult",
    "apply_perceptual_residual_guard",
    "apply_target_perceptual_residual_guard",
]
