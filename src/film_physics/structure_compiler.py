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


def _gaussian_kernel_1d(sigma: float, truncate: float) -> np.ndarray:
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Gaussian sigma must be positive")
    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    return kernel / np.sum(kernel, dtype=np.float64)


def zero_dc_dog_kernel_metrics(
    narrow_sigma: float,
    broad_sigma: float,
    *,
    broad_weight: float = 1.0,
    truncate: float = 4.0,
) -> tuple[float, float]:
    """Return the DC sum and variance of a 2D Gaussian-difference kernel."""

    if (
        not math.isfinite(broad_weight)
        or broad_weight <= 0.0
        or not math.isfinite(truncate)
        or truncate <= 0.0
        or broad_sigma <= narrow_sigma
    ):
        raise ValueError("invalid difference-of-Gaussians parameters")
    narrow = _gaussian_kernel_1d(narrow_sigma, truncate)
    broad = _gaussian_kernel_1d(broad_sigma, truncate)
    radius = max(narrow.size, broad.size) // 2

    def pad(kernel: np.ndarray) -> np.ndarray:
        result = np.zeros(2 * radius + 1, dtype=np.float64)
        start = radius - kernel.size // 2
        result[start : start + kernel.size] = kernel
        return result

    narrow = pad(narrow)
    broad = pad(broad)
    kernel = np.outer(narrow, narrow) - broad_weight * np.outer(broad, broad)
    dc_sum = float(np.sum(kernel, dtype=np.float64))
    variance = float(np.sum(np.square(kernel), dtype=np.float64))
    if not math.isfinite(variance) or variance <= 0.0:
        raise RuntimeError("invalid difference-of-Gaussians variance")
    return dc_sum, variance


def zero_dc_dog_normal_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    narrow_sigma: float,
    broad_sigma: float,
    seed: int,
    broad_weight: float = 1.0,
    truncate: float = 4.0,
) -> np.ndarray:
    """Return a coordinate-stable normalized Gaussian band-pass field."""

    full_height, full_width = full_shape
    origin_y, origin_x = origin_yx
    height, width = shape
    if not (
        0 <= origin_y < origin_y + height <= full_height
        and 0 <= origin_x < origin_x + width <= full_width
    ):
        raise ValueError("difference-of-Gaussians region is outside the full field")
    _, variance = zero_dc_dog_kernel_metrics(
        narrow_sigma,
        broad_sigma,
        broad_weight=broad_weight,
        truncate=truncate,
    )
    halo = int(truncate * broad_sigma + 0.5)
    y0 = max(0, origin_y - halo)
    x0 = max(0, origin_x - halo)
    y1 = min(full_height, origin_y + height + halo)
    x1 = min(full_width, origin_x + width + halo)
    white = counter_normal_region(
        full_shape,
        origin_yx=(y0, x0),
        shape=(y1 - y0, x1 - x0),
        seed=seed,
    )
    narrow = gaussian_filter(
        white,
        sigma=narrow_sigma,
        mode="constant",
        cval=0.0,
        truncate=truncate,
    )
    broad = gaussian_filter(
        white,
        sigma=broad_sigma,
        mode="constant",
        cval=0.0,
        truncate=truncate,
    )
    normalized = (narrow - broad_weight * broad) / math.sqrt(variance)
    crop_y = origin_y - y0
    crop_x = origin_x - x0
    return normalized[crop_y : crop_y + height, crop_x : crop_x + width]


def _balanced_gaussian_variance_scale(sigma: float, truncate: float = 4.0) -> float:
    """Exact infinite-grid variance after filtering zero-sum 2x2 normals."""

    if sigma <= 0.0:
        return 1.0
    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    kernel = np.exp(-0.5 * np.square(offsets.astype(np.float64) / sigma))
    kernel /= np.sum(kernel)
    blocks: dict[tuple[int, int], list[float]] = {}
    for iy, dy in enumerate(offsets):
        for ix, dx in enumerate(offsets):
            key = (int(dy) // 2, int(dx) // 2)
            blocks.setdefault(key, []).append(float(kernel[iy] * kernel[ix]))
    variance = sum(
        (4.0 / 3.0) * sum(weight * weight for weight in weights)
        - (1.0 / 3.0) * sum(weights) ** 2
        for weights in blocks.values()
    )
    if not math.isfinite(variance) or variance <= 0.0:
        raise RuntimeError("invalid balanced Gaussian variance")
    return variance


def balanced_correlated_normal_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Return a coordinate-stable Gaussian field with zero-sum 2x2 sources."""

    full_height, full_width = full_shape
    origin_y, origin_x = origin_yx
    height, width = shape
    if not (
        0 <= origin_y < origin_y + height <= full_height
        and 0 <= origin_x < origin_x + width <= full_width
        and math.isfinite(sigma)
        and sigma >= 0.0
    ):
        raise ValueError("balanced normal region is outside the full field")
    halo = int(4.0 * sigma + 0.5) if sigma > 0.0 else 0
    y0 = max(0, origin_y - halo)
    x0 = max(0, origin_x - halo)
    y1 = min(full_height, origin_y + height + halo)
    x1 = min(full_width, origin_x + width + halo)
    y0 -= y0 % 2
    x0 -= x0 % 2
    y1 = min(full_height, y1 + (y1 % 2))
    x1 = min(full_width, x1 + (x1 % 2))
    white = counter_normal_region(
        full_shape,
        origin_yx=(y0, x0),
        shape=(y1 - y0, x1 - x0),
        seed=seed,
    )
    padded_height = white.shape[0] + white.shape[0] % 2
    padded_width = white.shape[1] + white.shape[1] % 2
    padded = np.zeros((padded_height, padded_width), dtype=np.float64)
    mask = np.zeros_like(padded)
    padded[: white.shape[0], : white.shape[1]] = white
    mask[: white.shape[0], : white.shape[1]] = 1.0
    block = padded.reshape(padded_height // 2, 2, padded_width // 2, 2)
    block_mask = mask.reshape(padded_height // 2, 2, padded_width // 2, 2)
    count = np.sum(block_mask, axis=(1, 3), keepdims=True)
    mean = np.sum(block * block_mask, axis=(1, 3), keepdims=True) / count
    scale = np.zeros_like(count)
    active = count > 1.0
    scale[active] = np.sqrt(count[active] / (count[active] - 1.0))
    balanced = ((block - mean) * block_mask * scale).reshape(padded.shape)
    balanced = balanced[: white.shape[0], : white.shape[1]]
    if sigma > 0.0:
        balanced = gaussian_filter(
            balanced,
            sigma=sigma,
            order=0,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        ) / math.sqrt(_balanced_gaussian_variance_scale(sigma))
    crop_y = origin_y - y0
    crop_x = origin_x - x0
    return balanced[crop_y : crop_y + height, crop_x : crop_x + width]


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
