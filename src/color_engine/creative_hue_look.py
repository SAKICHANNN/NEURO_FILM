"""Private hue-selective creative grading; no stock or skin-identification claim."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


def _finite(value: object, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("parameter must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError("parameter outside declared range")
    return result


@dataclass(frozen=True)
class CreativeHueLook:
    tone: tuple[float, float]
    green_shift: float
    blue_shift: float
    green_logsat: float
    blue_logsat: float

    def __post_init__(self) -> None:
        if not isinstance(self.tone, (tuple, list)) or len(self.tone) != 2:
            raise ValueError("tone needs two ordered controls")
        tone = tuple(_finite(v, 0.05, 0.95) for v in self.tone)
        if tone[0] > tone[1]:
            raise ValueError("tone controls must be ordered")
        object.__setattr__(self, "tone", tone)
        for name in ("green_shift", "blue_shift"):
            object.__setattr__(self, name, _finite(getattr(self, name), -60, 60))
        for name in ("green_logsat", "blue_logsat"):
            object.__setattr__(self, name, _finite(getattr(self, name), -1, 1))


def _hue_weight(hue: np.ndarray, centre: float) -> np.ndarray:
    distance = np.abs((hue - centre + 180) % 360 - 180)
    # Clamping constructs a compact-support mask, not an output gamut repair.
    t = np.maximum(0.0, 1.0 - distance / 65.0)
    return t * t * (3.0 - 2.0 * t)


def render_creative_hue_look(
    encoded_srgb: np.ndarray, spec: CreativeHueLook, *, amount: float = 1.0
) -> np.ndarray:
    """Owned finite pointwise RGB; caller establishes display-sRGB identity.

    Smooth green/blue hue masks leave warm hue/saturation unchanged, but apply
    tone to every colour. This is not a skin detector. No statistics, learned
    assets, clipping of output, resampling or randomness.
    """
    if not isinstance(spec, CreativeHueLook):
        raise TypeError("invalid creative hue specification")
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
    safe_delta = np.where(delta > 0, delta, 1)
    hue = np.where(
        value == r,
        (g - b) / safe_delta,
        np.where(value == g, (b - r) / safe_delta + 2, (r - g) / safe_delta + 4),
    )
    hue = np.where(delta > 0, (hue * 60) % 360, 0)
    saturation = np.divide(delta, value, out=np.zeros_like(value), where=value > 0)
    green, blue = _hue_weight(hue, 140), _hue_weight(hue, 235)
    hue = (hue + spec.green_shift * green + spec.blue_shift * blue) % 360
    gain = np.exp(spec.green_logsat * green + spec.blue_logsat * blue)
    scaled = saturation * gain
    saturation = scaled / ((1 - saturation) + scaled)
    a, b_control = spec.tone
    value = (
        3 * a * (1 - value) ** 2 * value
        + 3 * b_control * (1 - value) * value**2
        + value**3
    )
    k = (hue[..., None] / 60 + np.array([5.0, 3.0, 1.0])) % 6
    shape = np.maximum(0, np.minimum(np.minimum(k, 4 - k), 1))
    graded = value[..., None] * (1 - saturation[..., None] * shape)
    result = ((1 - strength) * x + strength * graded).astype(np.float32)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("unexpected output domain violation")
    return np.ascontiguousarray(result)
