"""Counter-based Gaussian-copula approximation for developed structure."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import betaincinv, gammaincinv, ndtr


_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@dataclass(frozen=True)
class MarginalProfile:
    family: str
    mean: float
    variance: float
    correlation_sigma_pixels: float
    seed: int

    def __post_init__(self) -> None:
        if self.family not in {"gamma-density", "beta-transmittance"}:
            raise ValueError("unsupported structure marginal family")
        values = (self.mean, self.variance, self.correlation_sigma_pixels)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("marginal parameters must be finite")
        if self.mean <= 0.0 or self.variance <= 0.0:
            raise ValueError("marginal mean and variance must be positive")
        if self.correlation_sigma_pixels < 0.0:
            raise ValueError("correlation sigma must be nonnegative")
        if self.family == "beta-transmittance":
            if self.mean >= 1.0 or self.variance >= self.mean * (1.0 - self.mean):
                raise ValueError("beta mean/variance are outside the valid domain")
        if not isinstance(self.seed, int) or self.seed < 0 or self.seed >= 2**64:
            raise ValueError("marginal seed must be unsigned 64-bit")


def _splitmix64(values: np.ndarray) -> np.ndarray:
    state = (values + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
    state = ((state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
    state = ((state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
    return state ^ (state >> np.uint64(31))


def counter_normal_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    seed: int,
) -> np.ndarray:
    full_height, full_width = full_shape
    origin_y, origin_x = origin_yx
    height, width = shape
    if not (
        0 <= origin_y < origin_y + height <= full_height
        and 0 <= origin_x < origin_x + width <= full_width
    ):
        raise ValueError("counter region is outside the full field")
    ys = np.arange(origin_y, origin_y + height, dtype=np.uint64)[:, None]
    xs = np.arange(origin_x, origin_x + width, dtype=np.uint64)[None, :]
    counters = ys * np.uint64(full_width) + xs + np.uint64(seed)
    first = _splitmix64(counters)
    second = _splitmix64(counters ^ np.uint64(0xD2B74407B1CE6E93))
    u1 = ((first >> np.uint64(11)).astype(np.float64) + 0.5) / float(2**53)
    u2 = ((second >> np.uint64(11)).astype(np.float64) + 0.5) / float(2**53)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def counter_uniform_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    seed: int,
) -> np.ndarray:
    """Return an open-interval coordinate-stable uniform field."""
    full_height, full_width = full_shape
    origin_y, origin_x = origin_yx
    height, width = shape
    if not (
        0 <= origin_y < origin_y + height <= full_height
        and 0 <= origin_x < origin_x + width <= full_width
    ):
        raise ValueError("counter region is outside the full field")
    ys = np.arange(origin_y, origin_y + height, dtype=np.uint64)[:, None]
    xs = np.arange(origin_x, origin_x + width, dtype=np.uint64)[None, :]
    counters = ys * np.uint64(full_width) + xs + np.uint64(seed)
    values = _splitmix64(counters)
    return ((values >> np.uint64(11)).astype(np.float64) + 0.5) / float(2**53)


def _gaussian_variance_scale(sigma: float, truncate: float = 4.0) -> float:
    if sigma <= 0.0:
        return 1.0
    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    kernel /= np.sum(kernel)
    return float(np.sum(np.square(kernel)))


def correlated_normal_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    sigma: float,
    seed: int,
) -> np.ndarray:
    if sigma <= 0.0:
        return counter_normal_region(
            full_shape, origin_yx=origin_yx, shape=shape, seed=seed
        )
    halo = int(4.0 * sigma + 0.5)
    y0 = max(0, origin_yx[0] - halo)
    x0 = max(0, origin_yx[1] - halo)
    y1 = min(full_shape[0], origin_yx[0] + shape[0] + halo)
    x1 = min(full_shape[1], origin_yx[1] + shape[1] + halo)
    white = counter_normal_region(
        full_shape,
        origin_yx=(y0, x0),
        shape=(y1 - y0, x1 - x0),
        seed=seed,
    )
    blurred = gaussian_filter(
        white,
        sigma=sigma,
        order=0,
        mode="constant",
        cval=0.0,
        truncate=4.0,
    )
    variance = _gaussian_variance_scale(sigma) ** 2
    normalized = blurred / math.sqrt(variance)
    crop_y = origin_yx[0] - y0
    crop_x = origin_yx[1] - x0
    return normalized[crop_y : crop_y + shape[0], crop_x : crop_x + shape[1]]


def render_marginal_region(
    profile: MarginalProfile,
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    normal = correlated_normal_region(
        full_shape,
        origin_yx=origin_yx,
        shape=shape,
        sigma=profile.correlation_sigma_pixels,
        seed=profile.seed,
    )
    uniform = ndtr(normal)
    if np.any(uniform <= 0.0) or np.any(uniform >= 1.0):
        raise RuntimeError("Gaussian copula escaped the open unit interval")
    if profile.family == "gamma-density":
        shape_parameter = profile.mean * profile.mean / profile.variance
        scale = profile.variance / profile.mean
        values = gammaincinv(shape_parameter, uniform) * scale
    else:
        common = profile.mean * (1.0 - profile.mean) / profile.variance - 1.0
        alpha = profile.mean * common
        beta = (1.0 - profile.mean) * common
        values = betaincinv(alpha, beta, uniform)
    output = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output <= 0.0):
        raise RuntimeError("compiled marginal produced invalid samples")
    if profile.family == "beta-transmittance" and np.any(output >= 1.0):
        raise RuntimeError("compiled beta transmittance escaped (0, 1)")
    output.setflags(write=False)
    return output


def render_marginal(profile: MarginalProfile, shape: tuple[int, int]) -> np.ndarray:
    return render_marginal_region(
        profile, shape, origin_yx=(0, 0), shape=shape
    )
