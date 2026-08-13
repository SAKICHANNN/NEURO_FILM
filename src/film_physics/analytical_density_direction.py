"""Analytical unit-transmittance envelope for a developed-density direction."""

from __future__ import annotations

import numpy as np


def apply_analytical_density_direction(
    base_transmittance: np.ndarray,
    density_residual: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the largest per-pixel scalar that keeps all channels in (0, 1]."""

    base = np.asarray(base_transmittance, dtype=np.float64)
    residual = np.asarray(density_residual, dtype=np.float64)
    if (
        base.shape != residual.shape
        or base.ndim < 1
        or base.shape[-1] != 3
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(residual))
        or np.any(base <= 0.0)
        or np.any(base > 1.0)
    ):
        raise ValueError("invalid analytical density-direction input")
    channel_scale = np.ones_like(residual)
    brightening = residual < 0.0
    if np.any(brightening):
        headroom_density = -np.log10(base)
        channel_scale[brightening] = np.minimum(
            1.0, headroom_density[brightening] / -residual[brightening]
        )
    scale = np.min(channel_scale, axis=-1, keepdims=True)
    scale = np.maximum(scale, 0.0)
    limited = scale < 1.0
    if np.any(limited):
        scale[limited] *= 1.0 - 16.0 * np.finfo(np.float64).eps
    applied_density = scale * residual
    output64 = base * np.power(10.0, -applied_density)
    output = np.ascontiguousarray(output64, dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output <= 0.0) or np.any(output > 1.0):
        raise RuntimeError("analytical density direction escaped unit transmittance")
    nonzero = np.abs(residual) > np.finfo(np.float64).tiny
    realized = np.zeros_like(residual)
    realized[nonzero] = applied_density[nonzero] / residual[nonzero]
    direction_error = float(
        np.max(
            np.abs(realized[nonzero] - np.broadcast_to(scale, residual.shape)[nonzero]),
            initial=0.0,
        )
    )
    return output, {
        "minimum_density_direction_scale": float(np.min(scale)),
        "limited_fraction": float(np.mean(limited)),
        "maximum_density_direction_scale_error": direction_error,
        "maximum_output_transmittance": float(np.max(output)),
        "minimum_output_transmittance": float(np.min(output)),
        "hard_clipping_used": 0.0,
    }


__all__ = ["apply_analytical_density_direction"]
