"""Subpixel developed-density moment compiler and positive coarse renderer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import gammaincinv, ndtr

from .derivative_conditioned_structure import DerivativeConditionedStructureResult
from .structure_compiler import correlated_normal_region


@dataclass(frozen=True)
class CompiledDensityMoments:
    mean_density: np.ndarray
    variance_density: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_density)
        variance = np.asarray(self.variance_density)
        if (
            mean.dtype != np.float64
            or variance.dtype != np.float64
            or mean.shape != variance.shape
            or mean.ndim != 3
            or mean.shape[-1] != 3
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(variance))
            or np.any(mean <= 0.0)
            or np.any(variance < 0.0)
        ):
            raise ValueError("invalid compiled density moments")
        mean.setflags(write=False)
        variance.setflags(write=False)


def _gaussian_axis_correlation(sigma: float, factor: int, truncate: float) -> np.ndarray:
    if sigma == 0.0:
        return np.eye(factor, dtype=np.float64)
    radius = int(truncate * sigma + 0.5)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(coordinates / sigma))
    kernel /= np.sum(kernel, dtype=np.float64)
    autocorrelation = np.correlate(kernel, kernel, mode="full")
    center = kernel.size - 1
    offsets = np.arange(factor)[:, None] - np.arange(factor)[None, :]
    correlation = np.zeros((factor, factor), dtype=np.float64)
    valid = np.abs(offsets) <= center
    correlation[valid] = autocorrelation[center + offsets[valid]] / autocorrelation[center]
    return correlation


def gaussian_subpixel_covariance(
    correlation_sigma_pixels: float,
    pixel_size_factor: int,
    *,
    truncate: float = 4.0,
) -> np.ndarray:
    """Return fixed row-major subpixel covariance for one square output block."""

    sigma = float(correlation_sigma_pixels)
    if (
        not math.isfinite(sigma)
        or sigma < 0.0
        or not isinstance(pixel_size_factor, int)
        or pixel_size_factor < 1
        or pixel_size_factor > 64
        or not math.isfinite(truncate)
        or truncate <= 0.0
    ):
        raise ValueError("invalid subpixel covariance inputs")
    axis = _gaussian_axis_correlation(sigma, pixel_size_factor, truncate)
    covariance = np.kron(axis, axis)
    covariance.setflags(write=False)
    return covariance


def compile_subpixel_density_moments(
    target_density: np.ndarray,
    variance_density: np.ndarray,
    *,
    pixel_size_factor: int,
    correlation_sigma_pixels: float,
) -> CompiledDensityMoments:
    """Compile exact block means and delta-method correlated block variances."""

    mean = np.asarray(target_density, dtype=np.float64)
    variance = np.asarray(variance_density, dtype=np.float64)
    factor = pixel_size_factor
    if (
        mean.ndim != 3
        or mean.shape[-1] != 3
        or mean.shape != variance.shape
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(variance))
        or np.any(mean <= 0.0)
        or np.any(variance < 0.0)
        or not isinstance(factor, int)
        or factor < 1
        or mean.shape[0] % factor
        or mean.shape[1] % factor
    ):
        raise ValueError("invalid subpixel density moments")
    out_height, out_width = mean.shape[0] // factor, mean.shape[1] // factor
    count = factor * factor
    mean_blocks = mean.reshape(out_height, factor, out_width, factor, 3)
    compiled_mean = mean_blocks.mean(axis=(1, 3), dtype=np.float64)
    standard_deviation = np.sqrt(variance).reshape(out_height, factor, out_width, factor, 3)
    vectors = standard_deviation.transpose(0, 2, 4, 1, 3).reshape(out_height, out_width, 3, count)
    covariance = gaussian_subpixel_covariance(correlation_sigma_pixels, factor)
    compiled_variance = np.einsum("...i,ij,...j->...", vectors, covariance, vectors, optimize=True)
    compiled_variance /= float(count * count)
    return CompiledDensityMoments(
        np.asarray(compiled_mean, dtype=np.float64),
        np.asarray(compiled_variance, dtype=np.float64),
    )


def render_density_moment_structure_region(
    moments: CompiledDensityMoments,
    *,
    correlation_sigma_pixels: float,
    layer_seeds: tuple[int, int, int],
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> DerivativeConditionedStructureResult:
    full_shape = moments.mean_density.shape[:2]
    y0, x0 = origin_yx
    height, width = shape
    if (
        not math.isfinite(correlation_sigma_pixels)
        or correlation_sigma_pixels < 0.0
        or len(layer_seeds) != 3
        or y0 < 0
        or x0 < 0
        or height <= 0
        or width <= 0
        or y0 + height > full_shape[0]
        or x0 + width > full_shape[1]
    ):
        raise ValueError("invalid density-moment render region")
    mean = moments.mean_density[y0 : y0 + height, x0 : x0 + width]
    variance = moments.variance_density[y0 : y0 + height, x0 : x0 + width]
    output = np.empty_like(mean)
    for channel, seed in enumerate(layer_seeds):
        layer_mean = mean[..., channel]
        layer_variance = variance[..., channel]
        active = layer_variance > 0.0
        layer = layer_mean.copy()
        if np.any(active):
            normal = correlated_normal_region(
                full_shape,
                origin_yx=origin_yx,
                shape=shape,
                sigma=correlation_sigma_pixels,
                seed=seed,
            )
            uniform = ndtr(normal)
            gamma_shape = np.square(layer_mean[active]) / layer_variance[active]
            gamma_scale = layer_variance[active] / layer_mean[active]
            layer[active] = gammaincinv(gamma_shape, uniform[active]) * gamma_scale
        output[..., channel] = layer
    density = np.asarray(output, dtype=np.float32)
    transmittance = np.asarray(np.power(10.0, -density.astype(np.float64)), dtype=np.float32)
    return DerivativeConditionedStructureResult(density, transmittance)


def render_density_moment_structure(
    moments: CompiledDensityMoments,
    *,
    correlation_sigma_pixels: float,
    layer_seeds: tuple[int, int, int],
) -> DerivativeConditionedStructureResult:
    return render_density_moment_structure_region(
        moments,
        correlation_sigma_pixels=correlation_sigma_pixels,
        layer_seeds=layer_seeds,
        origin_yx=(0, 0),
        shape=moments.mean_density.shape[:2],
    )


__all__ = [
    "CompiledDensityMoments",
    "compile_subpixel_density_moments",
    "gaussian_subpixel_covariance",
    "render_density_moment_structure",
    "render_density_moment_structure_region",
]
