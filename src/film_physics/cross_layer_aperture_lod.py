"""Analytic effective kernels and exact block apertures for cloud LOD."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import convolve2d


def gaussian_aperture_effective_kernel(
    sigma: float, factor: int, *, truncate: float = 4.0
) -> np.ndarray:
    if (
        not math.isfinite(sigma)
        or sigma <= 0.0
        or not isinstance(factor, int)
        or factor <= 0
        or truncate <= 0.0
    ):
        raise ValueError("invalid Gaussian aperture kernel request")
    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * np.square(offsets / sigma))
    kernel_1d /= np.sum(kernel_1d)
    gaussian = np.outer(kernel_1d, kernel_1d)
    aperture = np.full((factor, factor), 1.0 / (factor * factor), dtype=np.float64)
    result = convolve2d(gaussian, aperture, mode="full")
    result /= np.sum(result)
    result.setflags(write=False)
    return result


def centered_kernel_inner_product(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    size_y, size_x = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])

    def pad(value: np.ndarray) -> np.ndarray:
        output = np.zeros((size_y, size_x), dtype=np.float64)
        y = (size_y - value.shape[0]) // 2
        x = (size_x - value.shape[1]) // 2
        output[y : y + value.shape[0], x : x + value.shape[1]] = value
        return output

    return float(np.sum(pad(a) * pad(b), dtype=np.float64))


def block_aperture_mean(values: np.ndarray, factor: int) -> np.ndarray:
    image = np.asarray(values)
    if (
        image.ndim != 3
        or image.shape[-1] != 3
        or image.shape[0] % factor
        or image.shape[1] % factor
    ):
        raise ValueError("image is incompatible with block aperture")
    result = image.reshape(
        image.shape[0] // factor, factor, image.shape[1] // factor, factor, 3
    ).mean(axis=(1, 3), dtype=np.float64)
    result.setflags(write=False)
    return result


__all__ = [
    "block_aperture_mean",
    "centered_kernel_inner_product",
    "gaussian_aperture_effective_kernel",
]
