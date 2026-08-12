"""Analytical no-clip composition of a physical linear-light residual."""

from __future__ import annotations

import numpy as np


def apply_bounded_linear_residual(
    neutral_base: np.ndarray,
    physical_scan: np.ndarray,
    cloud_free_scan: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    base = np.asarray(neutral_base, dtype=np.float64)
    physical = np.asarray(physical_scan, dtype=np.float64)
    reference = np.asarray(cloud_free_scan, dtype=np.float64)
    if (
        base.shape != physical.shape
        or base.shape != reference.shape
        or base.ndim < 1
        or base.shape[-1] != 3
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(physical))
        or not np.all(np.isfinite(reference))
        or np.any(base < 0.0)
        or np.any(base > 1.0)
    ):
        raise ValueError("invalid bounded linear residual input")
    residual = physical - reference
    scale = np.ones_like(residual)
    positive = residual > 0.0
    negative = residual < 0.0
    scale[positive] = np.minimum(1.0, (1.0 - base[positive]) / residual[positive])
    scale[negative] = np.minimum(1.0, -base[negative] / residual[negative])
    scale = np.maximum(scale, 0.0)
    limited = scale < 1.0
    scale[limited] *= 1.0 - 8.0 * np.finfo(np.float32).eps
    output = np.ascontiguousarray(base + scale * residual, dtype=np.float32)
    if np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("bounded linear residual escaped its output domain")
    return output, {
        "unbounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "bounded_residual_rms": float(
            np.sqrt(np.mean((output.astype(np.float64) - base) ** 2))
        ),
        "minimum_residual_scale": float(np.min(scale)),
        "limited_fraction": float(np.mean(limited)),
        "hard_clipping_used": 0.0,
    }


__all__ = ["apply_bounded_linear_residual"]
