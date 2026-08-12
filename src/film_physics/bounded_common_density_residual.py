"""Hue-ratio-preserving composition of one shared optical-density residual."""

from __future__ import annotations

import numpy as np


def apply_bounded_common_density_residual(
    neutral_base: np.ndarray,
    physical_scan: np.ndarray,
    cloud_free_scan: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project RGB scan differences to one common density field and apply it."""

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
        or np.any(physical <= 0.0)
        or np.any(reference <= 0.0)
    ):
        raise ValueError("invalid common-density residual input")

    layer_density_delta = -np.log10(physical / reference)
    common_density_delta = np.mean(layer_density_delta, axis=-1)
    log_gain = -np.log(10.0) * common_density_delta
    scale = np.ones_like(log_gain)
    brightening = log_gain > 0.0
    if np.any(brightening):
        positive_base = base > 0.0
        headroom = np.full_like(base, np.inf)
        headroom[positive_base] = np.log(
            (1.0 - 16.0 * np.finfo(np.float64).eps) / base[positive_base]
        )
        maximum_log_gain = np.min(headroom, axis=-1)
        scale[brightening] = np.minimum(
            1.0, maximum_log_gain[brightening] / log_gain[brightening]
        )
    scale = np.clip(scale, 0.0, 1.0)
    limited = scale < 1.0
    gain = np.exp(scale * log_gain)
    output64 = base * gain[..., None]
    output = np.ascontiguousarray(output64, dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("common-density residual escaped its output domain")
    residual = output.astype(np.float64) - base
    chromatic_density = layer_density_delta - common_density_delta[..., None]
    return output, {
        "unbounded_common_density_rms": float(
            np.sqrt(np.mean(common_density_delta * common_density_delta))
        ),
        "removed_chromatic_density_rms": float(
            np.sqrt(np.mean(chromatic_density * chromatic_density))
        ),
        "bounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "minimum_residual_scale": float(np.min(scale)),
        "limited_fraction": float(np.mean(limited)),
        "hard_clipping_used": 0.0,
    }


__all__ = ["apply_bounded_common_density_residual"]
