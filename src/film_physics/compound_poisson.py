"""Positive compound-Poisson structure fields in physical material domains."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from .structure_compiler import counter_uniform_region


@dataclass(frozen=True)
class CompoundPoissonProfile:
    family: str
    poisson_rate: float
    correlation_sigma_pixels: float
    baseline: float
    scale: float
    seed: int
    truncate: float = 4.0

    def __post_init__(self) -> None:
        if self.family not in {"density-shot", "transmittance-shot"}:
            raise ValueError("unsupported compound-Poisson output family")
        values = (
            self.poisson_rate,
            self.correlation_sigma_pixels,
            self.baseline,
            self.scale,
            self.truncate,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("compound-Poisson parameters must be finite")
        if (
            self.poisson_rate <= 0.0
            or self.poisson_rate > 64.0
            or self.scale <= 0.0
        ):
            raise ValueError("Poisson rate must be in (0, 64] and scale positive")
        if self.correlation_sigma_pixels < 0.0 or self.baseline < 0.0:
            raise ValueError("sigma and baseline must be nonnegative")
        if self.truncate <= 0.0:
            raise ValueError("Gaussian truncate must be positive")
        if self.family == "density-shot" and self.baseline <= 0.0:
            raise ValueError("density-shot baseline must be strictly positive")
        if not isinstance(self.seed, int) or self.seed < 0 or self.seed >= 2**64:
            raise ValueError("seed must be unsigned 64-bit")


def counter_poisson_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    rate: float,
    seed: int,
) -> np.ndarray:
    """Sample exact coordinate-stable Poisson counts by inverse recurrence."""
    if not math.isfinite(rate) or rate <= 0.0 or rate > 64.0:
        raise ValueError("Poisson rate must be finite and in (0, 64]")
    uniform = counter_uniform_region(
        full_shape, origin_yx=origin_yx, shape=shape, seed=seed
    )
    probability = np.full(shape, math.exp(-rate), dtype=np.float64)
    cumulative = probability.copy()
    counts = np.zeros(shape, dtype=np.uint16)
    active = uniform > cumulative
    order = 0
    while np.any(active):
        order += 1
        if order > 1024:
            raise RuntimeError("Poisson inverse recurrence did not converge")
        counts[active] += np.uint16(1)
        probability *= rate / order
        cumulative += probability
        active = uniform > cumulative
    counts.setflags(write=False)
    return counts


def _shot_field_region(
    profile: CompoundPoissonProfile,
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    sigma = profile.correlation_sigma_pixels
    if sigma <= 0.0:
        return counter_poisson_region(
            full_shape,
            origin_yx=origin_yx,
            shape=shape,
            rate=profile.poisson_rate,
            seed=profile.seed,
        ).astype(np.float64)
    halo = int(profile.truncate * sigma + 0.5)
    y0 = max(0, origin_yx[0] - halo)
    x0 = max(0, origin_yx[1] - halo)
    y1 = min(full_shape[0], origin_yx[0] + shape[0] + halo)
    x1 = min(full_shape[1], origin_yx[1] + shape[1] + halo)
    counts = counter_poisson_region(
        full_shape,
        origin_yx=(y0, x0),
        shape=(y1 - y0, x1 - x0),
        rate=profile.poisson_rate,
        seed=profile.seed,
    )
    filtered = gaussian_filter(
        counts.astype(np.float64),
        sigma=sigma,
        order=0,
        mode="constant",
        cval=0.0,
        truncate=profile.truncate,
    )
    crop_y = origin_yx[0] - y0
    crop_x = origin_yx[1] - x0
    return filtered[
        crop_y : crop_y + shape[0],
        crop_x : crop_x + shape[1],
    ]


def render_compound_poisson_region(
    profile: CompoundPoissonProfile,
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    shot = _shot_field_region(
        profile, full_shape, origin_yx=origin_yx, shape=shape
    )
    optical_density = profile.baseline + profile.scale * shot
    if profile.family == "density-shot":
        values = optical_density
    else:
        values = np.exp(-optical_density)
    output = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output <= 0.0):
        raise RuntimeError("compound-Poisson structure left its physical domain")
    if profile.family == "transmittance-shot" and np.any(output > 1.0):
        raise RuntimeError("compound-Poisson transmittance exceeds one")
    output.setflags(write=False)
    return output


def render_compound_poisson(
    profile: CompoundPoissonProfile, shape: tuple[int, int]
) -> np.ndarray:
    return render_compound_poisson_region(
        profile, shape, origin_yx=(0, 0), shape=shape
    )
