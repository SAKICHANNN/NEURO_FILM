"""Deterministic high-rate compound-Poisson measurement-cell sampling.

The independent-cell topology is a generic reference hypothesis.  This module
does not assert a measured film NPS, microscopic particle count, or renderer.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import pdtrik

from .structure_compiler import counter_uniform_region


def _counter_poisson_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    rate: float,
    seed: int,
    maximum_rate: float,
) -> np.ndarray:
    if not math.isfinite(rate) or rate < 1.0 or rate > maximum_rate:
        raise ValueError("Poisson rate is outside the supported sampler range")
    if not isinstance(seed, int) or seed < 0 or seed >= 2**64:
        raise ValueError("seed must be unsigned 64-bit")
    uniform = counter_uniform_region(
        full_shape, origin_yx=origin_yx, shape=shape, seed=seed
    )
    inverse = np.ceil(pdtrik(uniform, rate))
    if (
        not np.all(np.isfinite(inverse))
        or np.any(inverse < 0.0)
        or np.any(inverse > np.iinfo(np.uint32).max)
    ):
        raise RuntimeError("Poisson inverse left the uint32 count domain")
    counts = np.asarray(inverse, dtype=np.uint32)
    counts.setflags(write=False)
    return counts


def counter_high_rate_poisson_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    rate: float,
    seed: int,
) -> np.ndarray:
    """Sample the frozen P4BC high-rate regime by inverse CDF."""

    try:
        return _counter_poisson_region(
            full_shape,
            origin_yx=origin_yx,
            shape=shape,
            rate=rate,
            seed=seed,
            maximum_rate=1_000_000.0,
        )
    except ValueError as error:
        if "supported sampler range" in str(error):
            raise ValueError(
                "high-rate Poisson rate must be finite and in [1, 1e6]"
            ) from error
        raise


def counter_aperture_scaled_poisson_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    rate: float,
    seed: int,
) -> np.ndarray:
    """Sample the bounded P4BG multi-aperture rate range by inverse CDF."""

    return _counter_poisson_region(
        full_shape,
        origin_yx=origin_yx,
        shape=shape,
        rate=rate,
        seed=seed,
        maximum_rate=16_000_000.0,
    )


def sample_aperture_cell_density_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    rate: float,
    density_mark: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable counts and density for a 48 um cell region."""

    if not math.isfinite(density_mark) or density_mark <= 0.0:
        raise ValueError("density mark must be finite and positive")
    counts = counter_high_rate_poisson_region(
        full_shape, origin_yx=origin_yx, shape=shape, rate=rate, seed=seed
    )
    density = counts.astype(np.float64) * density_mark
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise RuntimeError("sampled cell density left its physical domain")
    density.setflags(write=False)
    return counts, density


def sample_aperture_scaled_density_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    rate: float,
    density_mark: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable counts and density for one P4BG-scaled aperture."""

    if not math.isfinite(density_mark) or density_mark <= 0.0:
        raise ValueError("density mark must be finite and positive")
    counts = counter_aperture_scaled_poisson_region(
        full_shape, origin_yx=origin_yx, shape=shape, rate=rate, seed=seed
    )
    density = counts.astype(np.float64) * density_mark
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise RuntimeError("sampled cell density left its physical domain")
    density.setflags(write=False)
    return counts, density


def sample_nonstationary_aperture_density_region(
    poisson_rate: np.ndarray,
    density_mark: np.ndarray,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one region from pointwise aperture-cell density parameters.

    The rate and mark arrays describe the complete coordinate field.  Region
    calls therefore retain the same counter coordinates as a full-field call.
    No realized-field moment correction or spatial filtering is performed.
    """

    rate = np.asarray(poisson_rate, dtype=np.float64)
    mark = np.asarray(density_mark, dtype=np.float64)
    if (
        rate.ndim != 2
        or rate.shape != mark.shape
        or not rate.size
        or not np.all(np.isfinite(rate))
        or not np.all(np.isfinite(mark))
        or np.any(rate < 1.0)
        or np.any(rate > 16_000_000.0)
        or np.any(mark <= 0.0)
    ):
        raise ValueError("nonstationary aperture parameters are invalid")
    if not isinstance(seed, int) or seed < 0 or seed >= 2**64:
        raise ValueError("seed must be unsigned 64-bit")
    origin_y, origin_x = origin_yx
    height, width = shape
    if (
        not isinstance(origin_y, int)
        or not isinstance(origin_x, int)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or origin_y < 0
        or origin_x < 0
        or height <= 0
        or width <= 0
        or origin_y + height > rate.shape[0]
        or origin_x + width > rate.shape[1]
    ):
        raise ValueError("nonstationary aperture region is invalid")
    uniform = counter_uniform_region(
        rate.shape, origin_yx=origin_yx, shape=shape, seed=seed
    )
    selected_rate = rate[
        origin_y : origin_y + height, origin_x : origin_x + width
    ]
    inverse = np.ceil(pdtrik(uniform, selected_rate))
    if (
        not np.all(np.isfinite(inverse))
        or np.any(inverse < 0.0)
        or np.any(inverse > np.iinfo(np.uint32).max)
    ):
        raise RuntimeError("Poisson inverse left the uint32 count domain")
    counts = np.asarray(inverse, dtype=np.uint32)
    density = counts.astype(np.float64) * mark[
        origin_y : origin_y + height, origin_x : origin_x + width
    ]
    if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
        raise RuntimeError("sampled nonstationary density left its physical domain")
    counts.setflags(write=False)
    density.setflags(write=False)
    return counts, density


__all__ = [
    "counter_aperture_scaled_poisson_region",
    "counter_high_rate_poisson_region",
    "sample_aperture_cell_density_region",
    "sample_aperture_scaled_density_region",
    "sample_nonstationary_aperture_density_region",
]
