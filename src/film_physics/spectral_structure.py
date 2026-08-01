"""Deterministic stationary Gaussian fields with an explicit nonnegative PSD."""

from __future__ import annotations

import math

import numpy as np

from .structure_compiler import counter_normal_region


def quantize_simplex_weights(
    weights: np.ndarray,
    *,
    denominator: int,
) -> np.ndarray:
    """Quantize nonnegative weights to integer units with an exact unit sum."""
    values = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise ValueError("invalid spectral simplex weights")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("spectral simplex weights have zero mass")
    scaled = values / total * denominator
    units = np.floor(scaled).astype(np.int64)
    missing = denominator - int(np.sum(units))
    if missing < 0 or missing >= len(values):
        raise RuntimeError("spectral weight remainder is invalid")
    remainders = scaled - units
    order = sorted(range(len(values)), key=lambda index: (-remainders[index], index))
    for index in order[:missing]:
        units[index] += 1
    if np.any(units < 0) or int(np.sum(units)) != denominator:
        raise RuntimeError("spectral weight quantization drift")
    output = np.ascontiguousarray(units / float(denominator), dtype=np.float64)
    output.setflags(write=False)
    return output


def log_gaussian_spectral_psd(
    shape: tuple[int, int],
    *,
    centers_cycles_per_pixel: np.ndarray,
    bandwidth_octaves: float,
    weights: np.ndarray,
    white_floor_fraction: float,
) -> np.ndarray:
    """Construct one nonnegative rFFT-domain radial power spectrum."""
    if len(shape) != 2 or min(shape) < 8:
        raise ValueError("spectral field shape is invalid")
    centers = np.asarray(centers_cycles_per_pixel, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if (
        centers.ndim != 1
        or values.shape != centers.shape
        or len(centers) < 2
        or not np.all(np.isfinite(centers))
        or not np.all(np.diff(centers) > 0.0)
        or centers[0] <= 0.0
        or centers[-1] >= math.sqrt(0.5)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not math.isclose(float(np.sum(values)), 1.0, abs_tol=1e-12)
        or not math.isfinite(bandwidth_octaves)
        or bandwidth_octaves <= 0.0
        or not math.isfinite(white_floor_fraction)
        or not 0.0 <= white_floor_fraction < 1.0
    ):
        raise ValueError("spectral mixture parameters are invalid")
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    safe_radius = np.maximum(radius, np.finfo(np.float64).tiny)
    mixture = np.zeros_like(radius)
    for center, weight in zip(centers, values, strict=True):
        distance = np.log2(safe_radius / center) / bandwidth_octaves
        mixture += weight * np.exp(-0.5 * np.square(distance))
    mixture = (1.0 - white_floor_fraction) * mixture + white_floor_fraction
    mixture[0, 0] = 0.0
    mean_power = float(np.mean(mixture))
    if mean_power <= 0.0 or not np.all(np.isfinite(mixture)):
        raise RuntimeError("spectral mixture is degenerate")
    output = np.ascontiguousarray(mixture / mean_power, dtype=np.float64)
    output.setflags(write=False)
    return output


def spectral_normal_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    centers_cycles_per_pixel: np.ndarray,
    bandwidth_octaves: float,
    weights: np.ndarray,
    white_floor_fraction: float,
    seed: int,
) -> np.ndarray:
    """Render an exact finite region from one coordinate-seeded spectral field."""
    height, width = full_shape
    y0, x0 = origin_yx
    region_height, region_width = shape
    if (
        height <= 0
        or width <= 0
        or y0 < 0
        or x0 < 0
        or region_height <= 0
        or region_width <= 0
        or y0 + region_height > height
        or x0 + region_width > width
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**64
    ):
        raise ValueError("invalid spectral region request")
    psd = log_gaussian_spectral_psd(
        full_shape,
        centers_cycles_per_pixel=centers_cycles_per_pixel,
        bandwidth_octaves=bandwidth_octaves,
        weights=weights,
        white_floor_fraction=white_floor_fraction,
    )
    innovation = counter_normal_region(
        full_shape,
        origin_yx=(0, 0),
        shape=full_shape,
        seed=seed,
    )
    transformed = np.fft.rfft2(innovation)
    field = np.fft.irfft2(transformed * np.sqrt(psd), s=full_shape)
    if not np.all(np.isfinite(field)):
        raise RuntimeError("spectral field is non-finite")
    output = np.ascontiguousarray(
        field[y0 : y0 + region_height, x0 : x0 + region_width],
        dtype=np.float64,
    )
    output.setflags(write=False)
    return output


__all__ = [
    "log_gaussian_spectral_psd",
    "quantize_simplex_weights",
    "spectral_normal_region",
]
