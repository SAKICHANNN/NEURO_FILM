"""Pixel-aperture and density-conditioned shared optical-density residual."""

from __future__ import annotations

import numpy as np

_KERNEL = np.asarray([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float64)


def _separable_aperture(values: np.ndarray) -> np.ndarray:
    padded_x = np.pad(values, ((0, 0), (2, 2)), mode="edge")
    horizontal = sum(
        weight * padded_x[:, offset : offset + values.shape[1]]
        for offset, weight in enumerate(_KERNEL)
    )
    padded_y = np.pad(horizontal, ((2, 2), (0, 0)), mode="edge")
    return sum(
        weight * padded_y[offset : offset + values.shape[0], :]
        for offset, weight in enumerate(_KERNEL)
    )


def apply_density_lod_residual(
    neutral_base: np.ndarray,
    physical_scan: np.ndarray,
    cloud_free_scan: np.ndarray,
    *,
    finite_tail_density: float | None = None,
    channel_density_gain: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Integrate one common density field over a pixel and gate by density."""

    base = np.asarray(neutral_base, dtype=np.float64)
    physical = np.asarray(physical_scan, dtype=np.float64)
    reference = np.asarray(cloud_free_scan, dtype=np.float64)
    if (
        base.shape != physical.shape
        or base.shape != reference.shape
        or base.ndim != 3
        or base.shape[-1] != 3
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(physical))
        or not np.all(np.isfinite(reference))
        or np.any(base < 0.0)
        or np.any(base > 1.0)
        or np.any(physical <= 0.0)
        or np.any(reference <= 0.0)
    ):
        raise ValueError("invalid density-LOD residual input")

    layer_density_delta = -np.log10(physical / reference)
    if channel_density_gain is not None:
        channel_gain = np.asarray(channel_density_gain, dtype=np.float64)
        if (
            channel_gain.shape != (3,)
            or not np.all(np.isfinite(channel_gain))
            or np.any(channel_gain <= 0.0)
        ):
            raise ValueError("invalid NPS-compiled channel density gain")
        layer_density_delta = layer_density_delta * channel_gain
    common_density_delta = np.mean(layer_density_delta, axis=-1)
    integrated_density_delta = _separable_aperture(common_density_delta)
    if finite_tail_density is not None:
        if not np.isfinite(finite_tail_density) or finite_tail_density <= 0.0:
            raise ValueError("invalid compound-Poisson finite-tail density")
        integrated_density_delta = finite_tail_density * np.tanh(
            integrated_density_delta / finite_tail_density
        )
    luminance = 0.2126 * base[..., 0] + 0.7152 * base[..., 1] + 0.0722 * base[..., 2]
    visibility = 4.0 * luminance * (1.0 - luminance)
    log_gain = -np.log(10.0) * integrated_density_delta * visibility

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
        raise RuntimeError("density-LOD residual escaped its output domain")

    residual = output.astype(np.float64) - base
    chromatic_density = layer_density_delta - common_density_delta[..., None]
    return output, {
        "unbounded_common_density_rms": float(
            np.sqrt(np.mean(common_density_delta * common_density_delta))
        ),
        "integrated_density_rms": float(
            np.sqrt(np.mean(integrated_density_delta * integrated_density_delta))
        ),
        "removed_chromatic_density_rms": float(
            np.sqrt(np.mean(chromatic_density * chromatic_density))
        ),
        "mean_density_visibility": float(np.mean(visibility)),
        "finite_tail_density": float(
            finite_tail_density if finite_tail_density is not None else np.inf
        ),
        "minimum_channel_density_gain": float(
            np.min(channel_density_gain) if channel_density_gain is not None else 1.0
        ),
        "bounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "minimum_residual_scale": float(np.min(scale)),
        "limited_fraction": float(np.mean(limited)),
        "hard_clipping_used": 0.0,
    }


__all__ = ["apply_density_lod_residual"]
