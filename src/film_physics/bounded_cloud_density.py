"""Analytical profile-envelope execution for stochastic density residuals."""

from __future__ import annotations

import numpy as np


def apply_bounded_cloud_density_residual(
    base_density: np.ndarray,
    cloud_density: np.ndarray,
    *,
    black_reference_density: np.ndarray,
    white_reference_density: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Scale a density residual to the exact profile envelope without clipping."""

    base = np.asarray(base_density, dtype=np.float64)
    candidate = np.asarray(cloud_density, dtype=np.float64)
    black = np.asarray(black_reference_density, dtype=np.float64)
    white = np.asarray(white_reference_density, dtype=np.float64)
    if (
        base.shape != candidate.shape
        or base.ndim < 1
        or base.shape[-1] != 3
        or black.shape != (3,)
        or white.shape != (3,)
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(black))
        or not np.all(np.isfinite(white))
        or np.any(white <= black)
        or np.any(base < black)
        or np.any(base > white)
    ):
        raise ValueError("invalid cloud density envelope input")
    residual = candidate - base
    scale = np.ones_like(residual)
    positive = residual > 0.0
    negative = residual < 0.0
    scale[positive] = np.minimum(
        1.0, (np.broadcast_to(white, base.shape)[positive] - base[positive]) / residual[positive]
    )
    scale[negative] = np.minimum(
        1.0, (np.broadcast_to(black, base.shape)[negative] - base[negative]) / residual[negative]
    )
    scale = np.maximum(scale, 0.0)
    limited = scale < 1.0
    scale[limited] *= 1.0 - 8.0 * np.finfo(np.float32).eps
    bounded64 = base + scale * residual
    bounded = np.ascontiguousarray(bounded64, dtype=np.float32)
    lower_violation = np.maximum(np.broadcast_to(black, bounded.shape) - bounded, 0.0)
    upper_violation = np.maximum(bounded - np.broadcast_to(white, bounded.shape), 0.0)
    maximum_violation = float(max(np.max(lower_violation), np.max(upper_violation)))
    if maximum_violation != 0.0:
        raise RuntimeError("analytical cloud density envelope escaped profile bounds")
    return bounded, {
        "unbounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "bounded_residual_rms": float(
            np.sqrt(np.mean((bounded.astype(np.float64) - base) ** 2))
        ),
        "minimum_residual_scale": float(np.min(scale)),
        "limited_fraction": float(np.mean(limited)),
        "maximum_density_envelope_violation": maximum_violation,
        "hard_clipping_used": 0.0,
    }


__all__ = ["apply_bounded_cloud_density_residual"]
