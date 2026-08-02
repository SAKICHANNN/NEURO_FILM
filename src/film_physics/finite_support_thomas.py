"""Finite-support separable approximation of a Thomas radial spectrum."""

from __future__ import annotations

import math

import numpy as np

from src.film_physics.structure_compiler import correlated_normal_region


def gaussian_kernel_1d(sigma: float, truncate: float) -> np.ndarray:
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(truncate) or truncate <= 0.0:
        raise ValueError("truncate must be finite and positive")
    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    kernel /= float(np.sum(kernel, dtype=np.float64))
    kernel.setflags(write=False)
    return kernel


def kernel_variance_2d(sigma: float, truncate: float) -> float:
    kernel = gaussian_kernel_1d(sigma, truncate)
    return float(np.sum(np.square(kernel), dtype=np.float64) ** 2)


def render_finite_support_thomas_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    particle_sigma_pixels: float,
    cluster_sigma_pixels: float,
    mean_offspring: float,
    component_seeds: tuple[int, int],
    realization_seed: int,
    truncate: float,
) -> np.ndarray:
    """Render one coordinate-stable unit-reference field region."""
    if (
        not math.isfinite(cluster_sigma_pixels)
        or cluster_sigma_pixels <= 0.0
        or not math.isfinite(mean_offspring)
        or mean_offspring <= 0.0
        or len(component_seeds) != 2
        or not isinstance(realization_seed, int)
        or realization_seed < 0
        or realization_seed >= 2**64
    ):
        raise ValueError("invalid Thomas finite-support profile")
    combined_sigma = math.hypot(particle_sigma_pixels, cluster_sigma_pixels)
    particle_variance = kernel_variance_2d(particle_sigma_pixels, truncate)
    combined_variance = kernel_variance_2d(combined_sigma, truncate)
    first = correlated_normal_region(
        full_shape,
        origin_yx=origin_yx,
        shape=shape,
        sigma=particle_sigma_pixels,
        seed=int(component_seeds[0]) ^ realization_seed,
    )
    second = correlated_normal_region(
        full_shape,
        origin_yx=origin_yx,
        shape=shape,
        sigma=combined_sigma,
        seed=int(component_seeds[1]) ^ realization_seed,
    )
    normalization = math.sqrt(particle_variance + mean_offspring * combined_variance)
    field = (
        math.sqrt(particle_variance) * first
        + math.sqrt(mean_offspring * combined_variance) * second
    ) / normalization
    output = np.ascontiguousarray(field, dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("finite-support Thomas field is non-finite")
    output.setflags(write=False)
    return output


__all__ = [
    "gaussian_kernel_1d",
    "kernel_variance_2d",
    "render_finite_support_thomas_region",
]
