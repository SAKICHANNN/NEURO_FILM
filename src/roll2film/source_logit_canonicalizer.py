"""Invertible source-only per-channel logit canonicalization."""

from __future__ import annotations

import numpy as np


def logit_shift(rgb: np.ndarray, shift: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    delta = np.asarray(shift, dtype=np.float64)
    if values.shape[-1] != 3 or delta.shape != (3,) or not np.all(np.isfinite(values)) or not np.all(np.isfinite(delta)):
        raise ValueError("invalid RGB or logit shift")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("RGB outside unit cube")
    output = values.copy()
    interior = (values > 0.0) & (values < 1.0)
    for channel in range(3):
        mask = interior[..., channel]
        x = values[..., channel][mask]
        z = np.log(x) - np.log1p(-x) + delta[channel]
        positive = z >= 0.0
        y = np.empty_like(z)
        y[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exponential = np.exp(z[~positive])
        y[~positive] = exponential / (1.0 + exponential)
        output[..., channel][mask] = y
    return output


def estimate_source_logit_shift(samples: np.ndarray, *, maximum_absolute_shift: float) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    maximum = float(maximum_absolute_shift)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("source samples must be finite Nx3")
    if np.any(values <= 0.0) or np.any(values >= 1.0) or maximum <= 0.0:
        raise ValueError("source samples must be strictly interior")
    logits = np.log(values) - np.log1p(-values)
    return np.clip(-np.median(logits, axis=0), -maximum, maximum)


__all__ = ["estimate_source_logit_shift", "logit_shift"]
