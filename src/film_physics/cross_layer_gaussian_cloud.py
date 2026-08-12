"""Gaussian cloud-footprint projection of cross-layer Poisson counts."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter


def render_cross_layer_gaussian_cloud_density(
    counts: np.ndarray,
    *,
    mark_optical_density_cmy: tuple[float, float, float],
    sigma_pixels_cmy: tuple[float, float, float],
    truncate: float = 4.0,
) -> np.ndarray:
    """Project C/M/Y event counts through normalized Gaussian cloud kernels."""

    values = np.asarray(counts)
    marks = np.asarray(mark_optical_density_cmy, dtype=np.float64)
    sigmas = np.asarray(sigma_pixels_cmy, dtype=np.float64)
    if (
        values.dtype != np.uint16
        or values.ndim != 3
        or values.shape[-1] != 3
        or marks.shape != (3,)
        or sigmas.shape != (3,)
        or np.any(marks <= 0.0)
        or np.any(sigmas <= 0.0)
        or not np.all(np.isfinite(marks))
        or not np.all(np.isfinite(sigmas))
        or not math.isfinite(truncate)
        or truncate <= 0.0
    ):
        raise ValueError("invalid cross-layer Gaussian cloud request")
    layers = [
        gaussian_filter(
            values[..., index].astype(np.float64),
            sigma=float(sigmas[index]),
            mode="wrap",
            truncate=truncate,
        )
        * marks[index]
        for index in range(3)
    ]
    density = np.stack(layers, axis=-1).astype(np.float32)
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise RuntimeError("Gaussian cloud density left its physical domain")
    density.setflags(write=False)
    return density


def discrete_gaussian_kernel_overlap(
    sigma_left: float, sigma_right: float, *, truncate: float = 4.0
) -> float:
    """Return exact discrete normalized-kernel inner product used by SciPy."""

    if min(sigma_left, sigma_right, truncate) <= 0.0 or not all(
        math.isfinite(value) for value in (sigma_left, sigma_right, truncate)
    ):
        raise ValueError("invalid Gaussian overlap request")
    radius = max(int(truncate * sigma_left + 0.5), int(truncate * sigma_right + 0.5))
    size = 2 * radius + 1
    impulse = np.zeros((size, size), dtype=np.float64)
    impulse[radius, radius] = 1.0
    left = gaussian_filter(
        impulse, sigma=sigma_left, mode="constant", cval=0.0, truncate=truncate
    )
    right = gaussian_filter(
        impulse, sigma=sigma_right, mode="constant", cval=0.0, truncate=truncate
    )
    return float(np.sum(left * right, dtype=np.float64))


__all__ = [
    "discrete_gaussian_kernel_overlap",
    "render_cross_layer_gaussian_cloud_density",
]
