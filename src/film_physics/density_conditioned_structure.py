"""Density-conditioned nonnegative marked-Poisson material structure."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from .structure_compiler import counter_uniform_region


DENSITY_CONDITIONED_STRUCTURE_SCHEMA = (
    "film_physics.density_conditioned_structure.v1"
)


@dataclass(frozen=True)
class DensityConditionedLayerProfile:
    layer_id: str
    grain_optical_density: float
    correlation_sigma_pixels: float
    seed: int
    maximum_target_density: float = 2.0
    truncate: float = 4.0

    def __post_init__(self) -> None:
        values = (
            self.grain_optical_density,
            self.correlation_sigma_pixels,
            self.maximum_target_density,
            self.truncate,
        )
        if (
            not self.layer_id
            or any(not math.isfinite(value) for value in values)
            or self.grain_optical_density <= 0.0
            or self.correlation_sigma_pixels < 0.0
            or self.maximum_target_density <= 0.0
            or self.truncate <= 0.0
            or self.maximum_target_density / self.grain_optical_density > 64.0
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.seed >= 2**64
        ):
            raise ValueError("invalid density-conditioned layer profile")


@dataclass(frozen=True)
class DensityConditionedStructureResult:
    density: np.ndarray
    transmittance: np.ndarray

    def __post_init__(self) -> None:
        density = np.asarray(self.density)
        transmittance = np.asarray(self.transmittance)
        if (
            density.dtype != np.float32
            or transmittance.dtype != np.float32
            or density.shape != transmittance.shape
            or density.ndim != 3
            or not np.all(np.isfinite(density))
            or not np.all(np.isfinite(transmittance))
            or np.any(density < 0.0)
            or np.any(transmittance <= 0.0)
            or np.any(transmittance > 1.0)
        ):
            raise ValueError("invalid density-conditioned structure result")
        density.setflags(write=False)
        transmittance.setflags(write=False)


def counter_poisson_rate_field(
    rate: np.ndarray,
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    seed: int,
) -> np.ndarray:
    """Sample coordinate-stable Poisson counts for a spatial rate field."""

    values = np.asarray(rate, dtype=np.float64)
    if (
        values.ndim != 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 64.0)
    ):
        raise ValueError("Poisson rate field must be finite in [0, 64]")
    uniform = counter_uniform_region(
        full_shape,
        origin_yx=origin_yx,
        shape=values.shape,
        seed=seed,
    )
    probability = np.exp(-values)
    cumulative = probability.copy()
    counts = np.zeros(values.shape, dtype=np.uint16)
    active = uniform > cumulative
    order = 0
    while np.any(active):
        order += 1
        if order > 1024:
            raise RuntimeError("variable-rate Poisson recurrence did not converge")
        counts[active] += np.uint16(1)
        probability *= values / order
        cumulative += probability
        active = uniform > cumulative
    counts.setflags(write=False)
    return counts


def _render_layer_region(
    target_density: np.ndarray,
    profile: DensityConditionedLayerProfile,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    full_shape = target_density.shape
    origin_y, origin_x = origin_yx
    height, width = shape
    if not (
        0 <= origin_y < origin_y + height <= full_shape[0]
        and 0 <= origin_x < origin_x + width <= full_shape[1]
    ):
        raise ValueError("density-conditioned region is outside the full field")
    sigma = profile.correlation_sigma_pixels
    halo = int(profile.truncate * sigma + 0.5) if sigma > 0.0 else 0
    y0 = max(0, origin_y - halo)
    x0 = max(0, origin_x - halo)
    y1 = min(full_shape[0], origin_y + height + halo)
    x1 = min(full_shape[1], origin_x + width + halo)
    density_region = target_density[y0:y1, x0:x1]
    rate = density_region / profile.grain_optical_density
    counts = counter_poisson_rate_field(
        rate,
        full_shape,
        origin_yx=(y0, x0),
        seed=profile.seed,
    ).astype(np.float64)
    if sigma > 0.0:
        counts = gaussian_filter(
            counts,
            sigma=sigma,
            order=0,
            mode="constant",
            cval=0.0,
            truncate=profile.truncate,
        )
    crop_y = origin_y - y0
    crop_x = origin_x - x0
    return (
        profile.grain_optical_density
        * counts[crop_y : crop_y + height, crop_x : crop_x + width]
    )


def render_density_conditioned_structure_region(
    target_density: np.ndarray,
    profiles: tuple[DensityConditionedLayerProfile, ...],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> DensityConditionedStructureResult:
    """Render one coordinate-stable region from developed optical density."""

    target = np.asarray(target_density, dtype=np.float64)
    if (
        target.ndim != 3
        or target.shape[2] != len(profiles)
        or not profiles
        or not np.all(np.isfinite(target))
        or np.any(target < 0.0)
    ):
        raise ValueError("target density must be finite nonnegative HxWxC")
    for channel, profile in enumerate(profiles):
        if np.any(target[..., channel] > profile.maximum_target_density):
            raise ValueError("target density exceeds the profile domain")
    layers = [
        _render_layer_region(
            target[..., channel],
            profile,
            origin_yx=origin_yx,
            shape=shape,
        )
        for channel, profile in enumerate(profiles)
    ]
    density = np.stack(layers, axis=-1).astype(np.float32)
    transmittance = np.exp(-density.astype(np.float64)).astype(np.float32)
    return DensityConditionedStructureResult(
        density=density,
        transmittance=transmittance,
    )


def render_density_conditioned_structure(
    target_density: np.ndarray,
    profiles: tuple[DensityConditionedLayerProfile, ...],
) -> DensityConditionedStructureResult:
    target = np.asarray(target_density)
    if target.ndim != 3:
        raise ValueError("target density must be HxWxC")
    return render_density_conditioned_structure_region(
        target,
        profiles,
        origin_yx=(0, 0),
        shape=target.shape[:2],
    )


__all__ = [
    "DENSITY_CONDITIONED_STRUCTURE_SCHEMA",
    "DensityConditionedLayerProfile",
    "DensityConditionedStructureResult",
    "counter_poisson_rate_field",
    "render_density_conditioned_structure",
    "render_density_conditioned_structure_region",
]
