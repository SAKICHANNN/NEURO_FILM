"""Deterministic tone/chroma factorization with source-inclusive gamut rails."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .positive_film import PositiveFilmResponseOperator


@dataclass(frozen=True)
class FactorizedBoundaryGuardResult:
    output: np.ndarray
    tone_scale: np.ndarray
    chroma_scale: np.ndarray


@dataclass(frozen=True)
class ResidualBoundaryGuardResult:
    output: np.ndarray
    residual_scale: np.ndarray


def _encoded_srgb_to_linear_scalar(value: float) -> float:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("encoded sRGB threshold must be finite and in [0,1]")
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _maximum_safe_scale(
    base: np.ndarray,
    delta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    positive = delta > 0.0
    negative = delta < 0.0
    positive_limit = np.where(
        positive, (upper - base) / np.maximum(delta, np.finfo(np.float64).tiny), np.inf
    )
    negative_limit = np.where(
        negative, (lower - base) / np.minimum(delta, -np.finfo(np.float64).tiny), np.inf
    )
    return np.clip(
        np.minimum(np.min(positive_limit, axis=-1), np.min(negative_limit, axis=-1)),
        0.0,
        1.0,
    )


def _source_inclusive_rails(
    source: np.ndarray,
    *,
    hard_boundary_epsilon_encoded_srgb: float,
    guard_boundary_epsilon_encoded_srgb: float,
) -> tuple[np.ndarray, np.ndarray]:
    hard_low = _encoded_srgb_to_linear_scalar(
        hard_boundary_epsilon_encoded_srgb
    )
    guard_low = _encoded_srgb_to_linear_scalar(
        guard_boundary_epsilon_encoded_srgb
    )
    hard_high = _encoded_srgb_to_linear_scalar(
        1.0 - hard_boundary_epsilon_encoded_srgb
    )
    guard_high = _encoded_srgb_to_linear_scalar(
        1.0 - guard_boundary_epsilon_encoded_srgb
    )
    lower = np.where(
        source <= hard_low,
        0.0,
        np.minimum(source, guard_low),
    )
    upper = np.where(
        source >= hard_high,
        1.0,
        np.maximum(source, guard_high),
    )
    return lower, upper


def apply_residual_boundary_guard(
    operator: PositiveFilmResponseOperator,
    linear_rgb: np.ndarray,
    *,
    strength: float,
    hard_boundary_epsilon_encoded_srgb: float,
    guard_boundary_epsilon_encoded_srgb: float,
) -> ResidualBoundaryGuardResult:
    """Apply the requested operator residual with analytical per-pixel scaling.

    The result follows the actual operator direction and scales it only where
    required to remain within source-inclusive encoded-sRGB guard rails.  This
    is deterministic gamut execution, not clipping or a per-image fit.
    """

    source = np.asarray(linear_rgb, dtype=np.float64)
    if (
        source.ndim < 2
        or source.shape[-1] != 3
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or not np.isfinite(strength)
        or strength < 0.0
        or strength > 1.0
        or not np.isfinite(hard_boundary_epsilon_encoded_srgb)
        or not np.isfinite(guard_boundary_epsilon_encoded_srgb)
        or hard_boundary_epsilon_encoded_srgb < 0.0
        or guard_boundary_epsilon_encoded_srgb
        <= hard_boundary_epsilon_encoded_srgb
        or guard_boundary_epsilon_encoded_srgb >= 0.5
    ):
        raise ValueError("invalid residual boundary-guard inputs")
    lower, upper = _source_inclusive_rails(
        source,
        hard_boundary_epsilon_encoded_srgb=(
            hard_boundary_epsilon_encoded_srgb
        ),
        guard_boundary_epsilon_encoded_srgb=(
            guard_boundary_epsilon_encoded_srgb
        ),
    )
    requested_delta = strength * (operator.apply(source) - source)
    residual_scale = _maximum_safe_scale(
        source, requested_delta, lower, upper
    )
    output = source + residual_scale[..., None] * requested_delta
    tolerance = 8.0 * np.finfo(np.float64).eps
    if (
        not np.all(np.isfinite(output))
        or np.any(output < lower - tolerance)
        or np.any(output > upper + tolerance)
    ):
        raise RuntimeError("residual boundary guard escaped source-inclusive rails")
    return ResidualBoundaryGuardResult(
        output=output,
        residual_scale=residual_scale,
    )


def apply_factorized_boundary_guard(
    operator: PositiveFilmResponseOperator,
    linear_rgb: np.ndarray,
    *,
    tone_strength: float,
    chroma_strength: float,
    luma_weights: np.ndarray,
    hard_boundary_epsilon_encoded_srgb: float,
    guard_boundary_epsilon_encoded_srgb: float,
) -> FactorizedBoundaryGuardResult:
    """Apply fixed operator residuals while preventing new sRGB boundary approach.

    The requested residual is split into an equal-RGB tone component carrying
    the operator's linear luminance delta and a zero-luminance chroma component.
    Each stage is scaled only as much as required to remain inside rails that
    always include the source pixel. No hard clipping is performed.
    """

    source = np.asarray(linear_rgb, dtype=np.float64)
    weights = np.asarray(luma_weights, dtype=np.float64)
    if (
        source.ndim < 2
        or source.shape[-1] != 3
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or weights.shape != (3,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or abs(float(np.sum(weights)) - 1.0) > 1e-12
        or not np.isfinite(tone_strength)
        or tone_strength < 0.0
        or tone_strength > 1.0
        or not np.isfinite(chroma_strength)
        or chroma_strength < 0.0
        or chroma_strength > 2.0
        or not np.isfinite(hard_boundary_epsilon_encoded_srgb)
        or not np.isfinite(guard_boundary_epsilon_encoded_srgb)
        or hard_boundary_epsilon_encoded_srgb < 0.0
        or guard_boundary_epsilon_encoded_srgb
        <= hard_boundary_epsilon_encoded_srgb
        or guard_boundary_epsilon_encoded_srgb >= 0.5
    ):
        raise ValueError("invalid factorized boundary-guard inputs")

    lower, upper = _source_inclusive_rails(
        source,
        hard_boundary_epsilon_encoded_srgb=(
            hard_boundary_epsilon_encoded_srgb
        ),
        guard_boundary_epsilon_encoded_srgb=(
            guard_boundary_epsilon_encoded_srgb
        ),
    )

    full = operator.apply(source)
    residual = full - source
    luma_delta = residual @ weights
    tone_delta = tone_strength * luma_delta[..., None]
    tone_scale = _maximum_safe_scale(source, tone_delta, lower, upper)
    tone_output = source + tone_scale[..., None] * tone_delta

    chroma_delta = chroma_strength * (
        residual - luma_delta[..., None]
    )
    chroma_scale = _maximum_safe_scale(
        tone_output, chroma_delta, lower, upper
    )
    output = tone_output + chroma_scale[..., None] * chroma_delta
    tolerance = 8.0 * np.finfo(np.float64).eps
    if (
        not np.all(np.isfinite(output))
        or np.any(output < lower - tolerance)
        or np.any(output > upper + tolerance)
    ):
        raise RuntimeError("factorized boundary guard escaped its source-inclusive rails")
    return FactorizedBoundaryGuardResult(
        output=output,
        tone_scale=tone_scale,
        chroma_scale=chroma_scale,
    )


__all__ = [
    "FactorizedBoundaryGuardResult",
    "ResidualBoundaryGuardResult",
    "apply_factorized_boundary_guard",
    "apply_residual_boundary_guard",
]
