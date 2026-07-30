"""HSV value/hue-saturation residuals with analytical RGB gamut guards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .factorized_boundary_guard import (
    _maximum_safe_scale,
    _source_inclusive_rails,
)
from .positive_film import PositiveFilmResponseOperator


@dataclass(frozen=True)
class HueValueResidualGuardResult:
    output: np.ndarray
    value_scale: np.ndarray
    hue_saturation_scale: np.ndarray


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    maximum = np.max(rgb, axis=-1)
    minimum = np.min(rgb, axis=-1)
    chroma = maximum - minimum
    saturation = np.divide(
        chroma,
        maximum,
        out=np.zeros_like(chroma),
        where=maximum > 0.0,
    )
    safe_chroma = np.where(chroma > 0.0, chroma, 1.0)
    red = np.mod((rgb[..., 1] - rgb[..., 2]) / safe_chroma, 6.0)
    green = (rgb[..., 2] - rgb[..., 0]) / safe_chroma + 2.0
    blue = (rgb[..., 0] - rgb[..., 1]) / safe_chroma + 4.0
    channel = np.argmax(rgb, axis=-1)
    hue_sector = np.choose(channel, [red, green, blue])
    hue = np.where(chroma > 0.0, hue_sector / 6.0, 0.0)
    return np.stack([hue, saturation, maximum], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    hue = np.mod(hsv[..., 0], 1.0)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    sector = hue * 6.0
    index = np.floor(sector).astype(np.int64)
    fraction = sector - np.floor(sector)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    choices = np.stack(
        [
            np.stack([value, t, p], axis=-1),
            np.stack([q, value, p], axis=-1),
            np.stack([p, value, t], axis=-1),
            np.stack([p, q, value], axis=-1),
            np.stack([t, p, value], axis=-1),
            np.stack([value, p, q], axis=-1),
        ],
        axis=-2,
    )
    return np.take_along_axis(
        choices, np.mod(index, 6)[..., None, None], axis=-2
    )[..., 0, :]


def apply_target_hue_value_residual_guard(
    linear_rgb: np.ndarray,
    target_linear_rgb: np.ndarray,
    *,
    value_strength: float,
    hue_saturation_strength: float,
    neutral_saturation_floor: float,
    hard_boundary_epsilon_encoded_srgb: float,
    guard_boundary_epsilon_encoded_srgb: float,
) -> HueValueResidualGuardResult:
    """Apply target value then hue/saturation residuals without clipping."""

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
        or not np.isfinite(value_strength)
        or value_strength < 0.0
        or value_strength > 1.0
        or not np.isfinite(hue_saturation_strength)
        or hue_saturation_strength < 0.0
        or hue_saturation_strength > 1.0
        or not np.isfinite(neutral_saturation_floor)
        or neutral_saturation_floor <= 0.0
        or neutral_saturation_floor > 0.25
    ):
        raise ValueError("invalid hue/value residual guard inputs")
    if np.array_equal(source, target):
        ones = np.ones(source.shape[:-1], dtype=np.float64)
        return HueValueResidualGuardResult(
            output=source.copy(),
            value_scale=ones,
            hue_saturation_scale=ones.copy(),
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
    source_hsv = _rgb_to_hsv(source)
    target_hsv = _rgb_to_hsv(target)

    requested_value_hsv = source_hsv.copy()
    requested_value_hsv[..., 2] += value_strength * (
        target_hsv[..., 2] - source_hsv[..., 2]
    )
    requested_value_rgb = _hsv_to_rgb(requested_value_hsv)
    value_delta = requested_value_rgb - source
    value_scale = _maximum_safe_scale(
        source, value_delta, lower, upper
    )
    value_output = source + value_scale[..., None] * value_delta

    value_output_hsv = _rgb_to_hsv(value_output)
    hue_delta = np.mod(
        target_hsv[..., 0] - source_hsv[..., 0] + 0.5, 1.0
    ) - 0.5
    source_hue_confidence = np.clip(
        source_hsv[..., 1] / neutral_saturation_floor,
        0.0,
        1.0,
    )
    target_hue_confidence = np.clip(
        target_hsv[..., 1] / neutral_saturation_floor,
        0.0,
        1.0,
    )
    hue_fraction = (
        hue_saturation_strength
        * np.minimum(source_hue_confidence, target_hue_confidence)
        + (1.0 - source_hue_confidence) * target_hue_confidence
    )
    requested_chroma_hsv = value_output_hsv.copy()
    requested_chroma_hsv[..., 0] = np.mod(
        value_output_hsv[..., 0]
        + hue_fraction * hue_delta,
        1.0,
    )
    requested_chroma_hsv[..., 1] = np.clip(
        value_output_hsv[..., 1]
        + hue_saturation_strength
        * (target_hsv[..., 1] - source_hsv[..., 1]),
        0.0,
        1.0,
    )
    requested_chroma_rgb = _hsv_to_rgb(requested_chroma_hsv)
    chroma_delta = requested_chroma_rgb - value_output
    hue_saturation_scale = _maximum_safe_scale(
        value_output, chroma_delta, lower, upper
    )
    output = (
        value_output
        + hue_saturation_scale[..., None] * chroma_delta
    )
    tolerance = 32.0 * np.finfo(np.float64).eps
    if (
        not np.all(np.isfinite(output))
        or np.any(output < lower - tolerance)
        or np.any(output > upper + tolerance)
    ):
        raise RuntimeError(
            "hue/value residual guard escaped source-inclusive rails"
        )
    return HueValueResidualGuardResult(
        output=output,
        value_scale=value_scale,
        hue_saturation_scale=hue_saturation_scale,
    )


def apply_hue_value_residual_guard(
    operator: PositiveFilmResponseOperator,
    linear_rgb: np.ndarray,
    **kwargs: object,
) -> HueValueResidualGuardResult:
    """Apply the hue/value factorization to one explicit film operator."""

    source = np.asarray(linear_rgb, dtype=np.float64)
    return apply_target_hue_value_residual_guard(
        source,
        operator.apply(source),
        **kwargs,
    )


__all__ = [
    "HueValueResidualGuardResult",
    "apply_hue_value_residual_guard",
    "apply_target_hue_value_residual_guard",
]
