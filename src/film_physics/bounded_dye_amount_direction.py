"""Shared-direction analytical envelope for normalized developed dye amounts."""

from __future__ import annotations

import numpy as np


def apply_bounded_dye_amount_direction(
    base_amount_rgb: np.ndarray,
    candidate_amount_rgb: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Keep one RGB residual direction while bounding every channel to [0, 1]."""

    base = np.asarray(base_amount_rgb, dtype=np.float64)
    candidate = np.asarray(candidate_amount_rgb, dtype=np.float64)
    if (
        base.shape != candidate.shape
        or base.ndim < 1
        or base.shape[-1] != 3
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(candidate))
        or np.any(base < 0.0)
        or np.any(base > 1.0)
    ):
        raise ValueError("invalid bounded dye-amount direction input")
    residual = candidate - base
    channel_scale = np.ones_like(residual)
    positive = residual > 0.0
    negative = residual < 0.0
    channel_scale[positive] = (1.0 - base[positive]) / residual[positive]
    channel_scale[negative] = -base[negative] / residual[negative]
    scale = np.minimum(1.0, np.min(channel_scale, axis=-1, keepdims=True))
    scale = np.maximum(scale, 0.0)
    limited = scale < 1.0
    scale[limited] = np.nextafter(scale[limited], 0.0)
    output = np.ascontiguousarray(base + scale * residual, dtype=np.float32)
    tolerance = 2.0 * np.finfo(np.float32).eps
    if (
        not np.all(np.isfinite(output))
        or np.any(output < -tolerance)
        or np.any(output > 1.0 + tolerance)
    ):
        raise RuntimeError("bounded dye amount escaped the unit cube")
    nonzero = np.abs(residual) > np.finfo(np.float64).tiny
    realized = np.zeros_like(residual)
    realized[nonzero] = (
        output.astype(np.float64)[nonzero] - base[nonzero]
    ) / residual[nonzero]
    expected = np.broadcast_to(scale, residual.shape)
    direction_error = float(np.max(np.abs(realized[nonzero] - expected[nonzero]), initial=0.0))
    return output, {
        "unbounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "bounded_residual_rms": float(
            np.sqrt(np.mean((output.astype(np.float64) - base) ** 2))
        ),
        "minimum_shared_scale": float(np.min(scale)),
        "limited_fraction": float(np.mean(limited)),
        "maximum_shared_direction_error": direction_error,
        "hard_clipping_used": 0.0,
    }


__all__ = ["apply_bounded_dye_amount_direction"]
