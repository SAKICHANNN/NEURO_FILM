"""Deterministic high-rate compound-Poisson measurement-cell sampling.

The independent-cell topology is a generic reference hypothesis.  This module
does not assert a measured film NPS, microscopic particle count, or renderer.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import pdtrik

from .structure_compiler import counter_uniform_region


def counter_high_rate_poisson_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    rate: float,
    seed: int,
) -> np.ndarray:
    """Sample coordinate-stable Poisson counts by inverse CDF.

    This vectorized inverse is intended for the high-rate P4BB measurement-cell
    regime, where the earlier small-rate recurrence is not computationally
    suitable.
    """

    if not math.isfinite(rate) or rate < 1.0 or rate > 1_000_000.0:
        raise ValueError("high-rate Poisson rate must be finite and in [1, 1e6]")
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


__all__ = [
    "counter_high_rate_poisson_region",
    "sample_aperture_cell_density_region",
]
