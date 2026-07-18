"""Safe Gaussian blur helpers for film-effect layers."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def _as_sigma_tuple(sigma: float | tuple[float, ...], ndim: int) -> tuple[float, ...]:
    if isinstance(sigma, tuple):
        if len(sigma) != ndim:
            raise ValueError(f"sigma length {len(sigma)} does not match ndim {ndim}")
        return tuple(max(0.0, float(value)) for value in sigma)
    return tuple(max(0.0, float(sigma)) for _ in range(ndim))


def _kernel1d(sigma: float, *, truncate: float) -> np.ndarray:
    if sigma <= 1e-6:
        return np.asarray([1.0], dtype=np.float32)
    radius = max(1, int(round(float(truncate) * float(sigma))))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel /= max(float(kernel.sum()), 1e-12)
    return kernel.astype(np.float32)


def _convolve_axis_reflect(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    if radius == 0:
        return array.astype(np.float32, copy=True)
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(array, pad_width, mode="reflect")
    moved = np.moveaxis(padded, axis, 0)
    out = np.empty((array.shape[axis],) + moved.shape[1:], dtype=np.float32)
    for index in range(array.shape[axis]):
        window = moved[index : index + len(kernel)]
        out[index] = np.tensordot(kernel, window, axes=(0, 0))
    return np.moveaxis(out, 0, axis).astype(np.float32, copy=False)


def _direct_gaussian_filter(array: np.ndarray, sigma: tuple[float, ...], *, truncate: float) -> np.ndarray:
    out = array.astype(np.float32, copy=False)
    for axis, axis_sigma in enumerate(sigma):
        if axis_sigma <= 1e-6:
            continue
        out = _convolve_axis_reflect(out, _kernel1d(axis_sigma, truncate=truncate), axis)
    return out.astype(np.float32, copy=False)


def _resize_channel(channel: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
    image = Image.fromarray(channel.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, resample=resample), dtype=np.float32)


def _resize_spatial(array: np.ndarray, shape: tuple[int, int], *, resample: int) -> np.ndarray:
    height, width = shape
    if array.ndim == 2:
        return _resize_channel(array, (width, height), resample)
    channels = [
        _resize_channel(array[..., channel], (width, height), resample)
        for channel in range(array.shape[2])
    ]
    return np.stack(channels, axis=2).astype(np.float32, copy=False)


def gaussian_filter_direct(
    array: np.ndarray,
    sigma: float | tuple[float, ...],
    *,
    truncate: float = 3.0,
) -> np.ndarray:
    """Apply the existing separable reflect Gaussian without resampling."""

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim not in {2, 3}:
        raise ValueError(f"gaussian_filter_direct expects 2D or 3D arrays, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("gaussian_filter_direct expects only finite values")
    if not math.isfinite(float(truncate)) or float(truncate) <= 0.0:
        raise ValueError("truncate must be finite and positive")
    sigma_tuple = _as_sigma_tuple(sigma, arr.ndim)
    channel_sigma = sigma_tuple[2] if arr.ndim == 3 else 0.0
    if channel_sigma > 1e-6:
        raise ValueError("gaussian_filter_direct does not support channel-axis blur")
    return _direct_gaussian_filter(arr, sigma_tuple, truncate=float(truncate))


def gaussian_filter_safe(
    array: np.ndarray,
    sigma: float | tuple[float, ...],
    *,
    truncate: float = 3.0,
    max_direct_radius: int = 32,
    target_downsampled_sigma: float = 6.0,
) -> np.ndarray:
    """Blur ``array`` with a bounded-cost Gaussian approximation.

    Small and medium kernels use direct separable convolution. Large spatial
    kernels are evaluated at lower resolution and upsampled. This is a deliberate
    approximation for low-frequency film-effect energy fields, not a replacement
    for scientific image analysis.
    """

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim not in {2, 3}:
        raise ValueError(f"gaussian_filter_safe expects 2D or 3D arrays, got {arr.shape}")
    sigma_tuple = _as_sigma_tuple(sigma, arr.ndim)
    if all(value <= 1e-6 for value in sigma_tuple):
        return arr.copy()

    spatial_sigma = sigma_tuple[:2]
    channel_sigma = sigma_tuple[2] if arr.ndim == 3 else 0.0
    if channel_sigma > 1e-6:
        raise ValueError("gaussian_filter_safe does not support channel-axis blur")

    max_radius = max(int(round(truncate * spatial_sigma[0])), int(round(truncate * spatial_sigma[1])))
    min_side = min(arr.shape[:2])
    if max_radius <= max_direct_radius or min_side <= 2:
        return _direct_gaussian_filter(arr, sigma_tuple, truncate=truncate)

    max_sigma = max(spatial_sigma)
    downsample = max(1, int(math.floor(max_sigma / max(target_downsampled_sigma, 1e-6))))
    downsample = min(downsample, max(1, min_side // 8))
    if downsample <= 1:
        return _direct_gaussian_filter(arr, sigma_tuple, truncate=truncate)

    low_shape = (max(2, arr.shape[0] // downsample), max(2, arr.shape[1] // downsample))
    low = _resize_spatial(arr, low_shape, resample=Image.Resampling.BOX)
    low_sigma = tuple(value / downsample for value in spatial_sigma)
    low_sigma_full = low_sigma + ((0.0,) if arr.ndim == 3 else ())
    blurred_low = _direct_gaussian_filter(low, low_sigma_full, truncate=truncate)
    return _resize_spatial(blurred_low, arr.shape[:2], resample=Image.Resampling.BILINEAR)
