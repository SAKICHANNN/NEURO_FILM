"""Bounded display-domain strength for the opt-in native Standard look."""

from __future__ import annotations

import numpy as np


def apply_native_standard_display_strength(
    source_encoded_srgb: np.ndarray,
    styled_encoded_srgb: np.ndarray,
    *,
    strength: float,
) -> np.ndarray:
    """Convexly scale a complete look in encoded display-sRGB.

    This is deliberately a display-domain Look Approximation control, not a
    physical exposure interpolation or calibrated film-response operation.
    """

    source = np.asarray(source_encoded_srgb)
    styled = np.asarray(styled_encoded_srgb)
    value = float(strength)
    if (
        source.dtype != np.float32
        or styled.dtype != np.float32
        or source.shape != styled.shape
        or source.ndim != 3
        or source.shape[-1] != 3
        or source.size == 0
        or not source.flags.c_contiguous
        or not styled.flags.c_contiguous
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(styled))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or np.any(styled < 0.0)
        or np.any(styled > 1.0)
        or not np.isfinite(value)
        or value < 0.0
        or value > 1.0
    ):
        raise ValueError(
            "native Standard strength requires finite contiguous float32 "
            "same-shape display-sRGB arrays and strength in [0,1]"
        )
    output = np.empty_like(source)
    np.subtract(styled, source, out=output)
    output *= np.float32(value)
    output += source
    if (
        not np.all(np.isfinite(output))
        or np.any(output < 0.0)
        or np.any(output > 1.0)
    ):
        raise RuntimeError("convex native Standard strength escaped gamut")
    return output


__all__ = ["apply_native_standard_display_strength"]
