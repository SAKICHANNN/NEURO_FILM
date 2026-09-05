"""Private authored print-like grade; encoded luma is not physical luminance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.color_engine.creative_look_v2 import _number, _vector


@dataclass(frozen=True)
class CreativePrintLook:
    tone: tuple[float, float] = (0.27, 0.90)
    chroma: float = 0.88
    shadow: tuple[float, float, float] = (-0.10, 0.015, 0.12)
    highlight: tuple[float, float, float] = (0.16, 0.025, -0.14)

    def __post_init__(self):
        t = _vector(self.tone, 2, 0.05, 0.95, "tone")
        if t[0] >= t[1]:
            raise ValueError("tone controls must strictly increase")
        object.__setattr__(self, "tone", t)
        object.__setattr__(self, "chroma", _number(self.chroma, 0, 1.5, "chroma"))
        for name in ("shadow", "highlight"):
            object.__setattr__(
                self, name, _vector(getattr(self, name), 3, -0.3, 0.3, name)
            )


def _luma(x):
    return x[..., 0:1] * 0.2126 + x[..., 1:2] * 0.7152 + x[..., 2:3] * 0.0722


def render_print_look(source: np.ndarray, spec: CreativePrintLook, *, amount=1.0):
    """Pointwise owned float32 grade without image statistics or RGB clipping.

    Scalar positive-derivative Bezier maps encoded luma; split tint is projected
    onto its zero-luma plane. Radial gamut compression retains that luma and
    chroma direction. This is a creative SDR map, not an invertibility, semantic
    skin protection, perceptual brightness or photographic safety guarantee.
    """
    if not isinstance(spec, CreativePrintLook):
        raise TypeError("invalid print look")
    strength = _number(amount, 0, 1, "amount")
    if (
        not isinstance(source, np.ndarray)
        or source.dtype != np.float32
        or source.ndim != 3
        or source.shape[-1] != 3
        or not source.size
        or not np.isfinite(source).all()
        or np.any((source < 0) | (source > 1))
    ):
        raise ValueError("requires bounded finite HxWx3 float32 encoded sRGB")
    if strength == 0:
        return source.copy(order="C")
    x = source.astype(np.float64)
    y = _luma(x)
    a, b = spec.tone
    t = 3 * a * (1 - y) ** 2 * y + 3 * b * (1 - y) * y**2 + y**3
    # Smooth complementary zones and zero tint at black/white, not hue masks.
    high = y * y * (3 - 2 * y)
    tint = (1 - high) * np.asarray(spec.shadow) + high * np.asarray(spec.highlight)
    tint = tint - _luma(tint)
    c = spec.chroma * (x - y) + 4 * y * (1 - y) * tint
    # One scale for the complete chroma vector, never independent RGB clamps.
    positive = np.max(c, axis=-1, keepdims=True)
    negative = -np.min(c, axis=-1, keepdims=True)
    required = np.maximum(
        np.divide(positive, 1 - t, out=np.zeros_like(t), where=t < 1),
        np.divide(negative, t, out=np.zeros_like(t), where=t > 0),
    )
    scale = 1 / np.maximum(1, required)
    # Floating-point guard only on compressed rays; not a post-output repair.
    scale = np.where(required > 1, scale * (1 - 1e-7), scale)
    candidate = t + c * scale
    result = np.ascontiguousarray(
        (1 - strength) * x + strength * candidate, dtype=np.float32
    )
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("print grade domain violation; no output clipping")
    return result
