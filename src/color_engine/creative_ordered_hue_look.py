"""Private ordered colour-design kernel; no stock or perceptual-luminance claim."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.color_engine.creative_hue_look import _finite, _hue_weight


@dataclass(frozen=True)
class OrderedHueLook:
    source_hue: tuple[float, ...]
    mapped_hue: tuple[float, ...]
    green_logsat: float = -0.4
    blue_logsat: float = -0.15
    value_lift: float = 0.0

    def __post_init__(self) -> None:
        for name in ("source_hue", "mapped_hue"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not 3 <= len(values) <= 32:
                raise ValueError("hue knots require 3..32 ordered entries")
            values = tuple(_finite(v, 0, 360) for v in values)
            if values[0] != 0 or values[-1] != 360 or np.any(np.diff(values) <= 0):
                raise ValueError("hue knots must strictly increase from 0 to 360")
            object.__setattr__(self, name, values)
        if len(self.source_hue) != len(self.mapped_hue):
            raise ValueError("hue knot lengths differ")
        slopes = np.diff(self.mapped_hue) / np.diff(self.source_hue)
        if np.any((slopes < 0.25) | (slopes > 3)):
            raise ValueError("hue secants outside [0.25,3]")
        for name in ("green_logsat", "blue_logsat"):
            object.__setattr__(self, name, _finite(getattr(self, name), -1, 1))
        object.__setattr__(self, "value_lift", _finite(self.value_lift, 0, 0.5))


def map_ordered_hue(
    hue: np.ndarray, spec: OrderedHueLook, *, amount: float = 1
) -> np.ndarray:
    """C1 periodic-end-slope Hermite map, strictly positive derivative.

    With d a segment secant and m0,m1 <= d, the derivative has quadratic
    Bernstein coefficients (m0, 3*d-m0-m1, m1), all positive. Convex strength
    interpolation with identity preserves this property.
    """
    if not isinstance(spec, OrderedHueLook):
        raise TypeError("invalid ordered hue specification")
    strength = _finite(amount, 0, 1)
    if (
        not isinstance(hue, np.ndarray)
        or not np.isfinite(hue).all()
        or np.any((hue < 0) | (hue > 360))
    ):
        raise ValueError("hue requires finite degrees in [0,360]")
    x, y = np.asarray(spec.source_hue), np.asarray(spec.mapped_hue)
    secant = np.diff(y) / np.diff(x)
    seam = min(secant[0], secant[-1])
    slope = np.r_[seam, np.minimum(secant[:-1], secant[1:]), seam]
    index = np.minimum(np.searchsorted(x, hue, side="right") - 1, len(x) - 2)
    width = x[index + 1] - x[index]
    t = (hue - x[index]) / width
    # Positive ordered Bezier controls implement the Hermite segment.
    b0, b3 = y[index], y[index + 1]
    b1 = b0 + width * slope[index] / 3
    b2 = b3 - width * slope[index + 1] / 3
    mapped = (
        (1 - t) ** 3 * b0
        + 3 * (1 - t) ** 2 * t * b1
        + 3 * (1 - t) * t * t * b2
        + t**3 * b3
    )
    return (1 - strength) * hue + strength * mapped


def render_ordered_hue_look(
    encoded_srgb: np.ndarray, spec: OrderedHueLook, *, amount: float = 1
) -> np.ndarray:
    """Owned bounded float32 RGB; optional monotone HSV-value lift.

    This version deliberately leaves the historical hue/tone kernel untouched.
    HSV conversion is explicit float64; one final float32 cast, no RGB clamp.
    Zero lift retains HSV value, not physical luminance. Lift v+k*v*(1-v)
    retains endpoints and has derivative >= 1-k >= 0.5.
    """
    if not isinstance(spec, OrderedHueLook):
        raise TypeError("invalid ordered hue specification")
    strength = _finite(amount, 0, 1)
    if (
        not isinstance(encoded_srgb, np.ndarray)
        or encoded_srgb.dtype != np.float32
        or encoded_srgb.ndim != 3
        or encoded_srgb.shape[-1] != 3
        or not encoded_srgb.size
        or not np.isfinite(encoded_srgb).all()
        or np.any((encoded_srgb < 0) | (encoded_srgb > 1))
    ):
        raise ValueError("requires finite HxWx3 float32 display-sRGB in [0,1]")
    if strength == 0:
        return encoded_srgb.copy(order="C")
    x = encoded_srgb.astype(np.float64)
    r, g, b = np.moveaxis(x, -1, 0)
    value = x.max(axis=-1)
    delta = value - x.min(axis=-1)
    safe = np.where(delta > 0, delta, 1)
    hue = np.where(
        value == r,
        (g - b) / safe,
        np.where(value == g, (b - r) / safe + 2, (r - g) / safe + 4),
    )
    hue = np.where(delta > 0, (hue * 60) % 360, 0)
    saturation = np.divide(delta, value, out=np.zeros_like(value), where=value > 0)
    gain = np.exp(
        strength
        * (
            spec.green_logsat * _hue_weight(hue, 105, 55)
            + spec.blue_logsat * _hue_weight(hue, 235, 45)
        )
    )
    scaled = saturation * gain
    saturation = scaled / (1 - saturation + scaled)
    if spec.value_lift:
        value = value + strength * spec.value_lift * value * (1 - value)
    mapped = map_ordered_hue(hue, spec, amount=strength) % 360
    k = (mapped[..., None] / 60 + np.array([5.0, 3.0, 1.0])) % 6
    shape = np.maximum(0, np.minimum(np.minimum(k, 4 - k), 1))
    result = (value[..., None] * (1 - saturation[..., None] * shape)).astype(np.float32)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("unexpected output domain violation")
    return np.ascontiguousarray(result)
