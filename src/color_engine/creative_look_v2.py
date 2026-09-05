"""Private, original creative Looks in encoded display-sRGB.

This is not a film response or a promoted product operator. No statistics,
learned weights, image-wide adaptation, resampling or final clipping is used.
Use bounded tiles for photographic-sized inputs; the map is pointwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


class CreativeLookError(ValueError):
    """Invalid display input or creative-operator parameters."""


def _number(value: object, low: float, high: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CreativeLookError(f"{name} must be a finite number in [{low}, {high}]")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise CreativeLookError(f"{name} must be a finite number in [{low}, {high}]")
    return result


def _vector(
    value: object, size: int, low: float, high: float, name: str
) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise CreativeLookError(f"{name} must have exactly {size} components")
    return tuple(_number(item, low, high, name) for item in value)


@dataclass(frozen=True)
class CreativeLookV2:
    """Immutable parameters, independent of names, files and product schemas."""

    tone: tuple[float, float]
    shadow: tuple[float, float, float]
    highlight: tuple[float, float, float]
    chroma: float

    def __post_init__(self) -> None:
        tone = _vector(self.tone, 2, 0.05, 0.95, "tone")
        if tone[0] > tone[1]:
            raise CreativeLookError("tone controls must be ordered")
        object.__setattr__(self, "tone", tone)
        object.__setattr__(self, "shadow", _vector(self.shadow, 3, -0.5, 0.5, "shadow"))
        object.__setattr__(
            self, "highlight", _vector(self.highlight, 3, -0.5, 0.5, "highlight")
        )
        object.__setattr__(self, "chroma", _number(self.chroma, -1.0, 1.5, "chroma"))


def render_creative_look_v2(
    encoded_srgb: np.ndarray, spec: CreativeLookV2, *, amount: float = 1.0
) -> np.ndarray:
    """Return owned bounded float32 RGB; zero amount is exact identity.

    The caller must establish encoded display-sRGB colour identity. This
    direct-module primitive cannot infer it from untagged numeric arrays.
    Global Bezier tone and smooth luma-zone gain operate independently at
    each pixel. The positive-gain rational map preserves [0,1] intrinsically;
    it does not prove photographic quality or quantized boundary safety.
    """

    if not isinstance(spec, CreativeLookV2):
        raise CreativeLookError("spec must be CreativeLookV2")
    strength = _number(amount, 0.0, 1.0, "amount")
    if (
        not isinstance(encoded_srgb, np.ndarray)
        or encoded_srgb.dtype != np.float32
        or encoded_srgb.ndim != 3
        or encoded_srgb.shape[2] != 3
        or encoded_srgb.size == 0
        or not np.isfinite(encoded_srgb).all()
        or np.any((encoded_srgb < 0.0) | (encoded_srgb > 1.0))
    ):
        raise CreativeLookError(
            "input must be finite bounded HxWx3 float32 display-sRGB"
        )
    if strength == 0.0:
        return encoded_srgb.copy(order="C")

    x = encoded_srgb.astype(np.float64)
    inv = 1.0 - x
    a, b = spec.tone
    toned = 3.0 * a * inv * inv * x + 3.0 * b * inv * x * x + x * x * x
    # Explicit three-term sum avoids shape-dependent BLAS/reduction choices.
    luma = (toned[..., 0] * 0.2126 + toned[..., 1] * 0.7152 + toned[..., 2] * 0.0722)[
        ..., None
    ]
    log_gain = (
        (1.0 - luma) ** 2 * np.asarray(spec.shadow)
        + luma**2 * np.asarray(spec.highlight)
        + spec.chroma * (toned - luma)
    )
    gain = np.exp(log_gain)
    candidate = toned * gain / (1.0 + toned * (gain - 1.0))
    result = np.ascontiguousarray(
        (1.0 - strength) * x + strength * candidate, dtype=np.float32
    )
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise CreativeLookError(
            "creative operator produced invalid output; no clipping"
        )
    return result
