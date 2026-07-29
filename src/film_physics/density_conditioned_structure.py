"""Density-conditioned nonnegative marked-Poisson material structure."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    maximum_filter,
    minimum_filter,
)
from scipy.signal import convolve2d

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
    count_rate_density: float | None = None
    boundary_mode: str = "zero-fill-v1"
    maximum_target_density: float = 2.0
    truncate: float = 4.0

    def __post_init__(self) -> None:
        rate_density = (
            self.grain_optical_density
            if self.count_rate_density is None
            else self.count_rate_density
        )
        values = (
            self.grain_optical_density,
            self.correlation_sigma_pixels,
            rate_density,
            self.maximum_target_density,
            self.truncate,
        )
        if (
            not self.layer_id
            or any(not math.isfinite(value) for value in values)
            or self.grain_optical_density <= 0.0
            or rate_density <= 0.0
            or self.correlation_sigma_pixels < 0.0
            or self.maximum_target_density <= 0.0
            or self.truncate <= 0.0
            or self.boundary_mode
            not in {"zero-fill-v1", "normalized-support-v1"}
            or self.maximum_target_density / rate_density > 4096.0
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.seed >= 2**64
        ):
            raise ValueError("invalid density-conditioned layer profile")

    @property
    def resolved_count_rate_density(self) -> float:
        return (
            self.grain_optical_density
            if self.count_rate_density is None
            else self.count_rate_density
        )


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
    maximum_rate: float | None = None,
) -> np.ndarray:
    """Sample coordinate-stable Poisson counts for a spatial rate field."""

    values = np.asarray(rate, dtype=np.float64)
    if (
        values.ndim != 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 4096.0)
    ):
        raise ValueError("Poisson rate field must be finite in [0, 4096]")
    observed_maximum = float(np.max(values, initial=0.0))
    decomposition_maximum = (
        observed_maximum if maximum_rate is None else float(maximum_rate)
    )
    if (
        not math.isfinite(decomposition_maximum)
        or decomposition_maximum < observed_maximum
        or decomposition_maximum > 4096.0
    ):
        raise ValueError("Poisson decomposition maximum is invalid")
    components = max(1, int(math.ceil(decomposition_maximum / 64.0)))
    if components > 1:
        component_rate = values / float(components)
        total = np.zeros(values.shape, dtype=np.uint32)
        for component in range(components):
            component_seed = (
                seed + component * 0x9E3779B97F4A7C15
            ) % (2**64)
            total += counter_poisson_rate_field(
                component_rate,
                full_shape,
                origin_yx=origin_yx,
                seed=component_seed,
                maximum_rate=decomposition_maximum / float(components),
            ).astype(np.uint32)
        if np.any(total > np.iinfo(np.uint16).max):
            raise RuntimeError("Poisson superposition exceeds uint16")
        output = total.astype(np.uint16)
        output.setflags(write=False)
        return output
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


def compile_density_conditioned_profiles(
    profiles: tuple[DensityConditionedLayerProfile, ...],
    *,
    pixel_size_factor: int,
    seed_offset: int = 0,
) -> tuple[DensityConditionedLayerProfile, ...]:
    """Compile base-pitch grains to area-mean output pixels."""

    if not isinstance(pixel_size_factor, int) or pixel_size_factor < 1:
        raise ValueError("pixel_size_factor must be a positive integer")
    if not isinstance(seed_offset, int):
        raise ValueError("seed_offset must be an integer")
    area = float(pixel_size_factor * pixel_size_factor)
    return tuple(
        DensityConditionedLayerProfile(
            layer_id=profile.layer_id,
            grain_optical_density=profile.grain_optical_density / area,
            correlation_sigma_pixels=(
                profile.correlation_sigma_pixels / pixel_size_factor
            ),
            seed=(profile.seed + seed_offset) % (2**64),
            count_rate_density=None,
            boundary_mode=profile.boundary_mode,
            maximum_target_density=profile.maximum_target_density,
            truncate=profile.truncate,
        )
        for profile in profiles
    )


def _discrete_gaussian_kernel_2d(
    sigma: float, truncate: float
) -> np.ndarray:
    if sigma == 0.0:
        return np.ones((1, 1), dtype=np.float64)
    radius = int(truncate * sigma + 0.5)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * (coordinates / sigma) ** 2)
    kernel_1d /= np.sum(kernel_1d, dtype=np.float64)
    return np.outer(kernel_1d, kernel_1d)


def effective_mark_loss(
    grain_optical_density: float,
    correlation_sigma_pixels: float,
    truncate: float,
) -> float:
    """Return the Poisson mark's analytic Beer-Lambert log loss."""

    values = (
        float(grain_optical_density),
        float(correlation_sigma_pixels),
        float(truncate),
    )
    if (
        any(not math.isfinite(value) for value in values)
        or grain_optical_density <= 0.0
        or correlation_sigma_pixels < 0.0
        or truncate <= 0.0
    ):
        raise ValueError("invalid effective-mark-loss parameters")
    kernel = _discrete_gaussian_kernel_2d(
        correlation_sigma_pixels, truncate
    )
    return float(
        np.sum(-np.expm1(-grain_optical_density * kernel), dtype=np.float64)
    )


def compile_effective_mark_loss_profiles(
    profiles: tuple[DensityConditionedLayerProfile, ...],
    *,
    pixel_size_factor: int,
    seed_offset: int = 0,
) -> tuple[DensityConditionedLayerProfile, ...]:
    """Compile profiles while preserving expected area transmittance."""

    compiled = compile_density_conditioned_profiles(
        profiles,
        pixel_size_factor=pixel_size_factor,
        seed_offset=seed_offset,
    )
    output = []
    for base, candidate in zip(profiles, compiled, strict=True):
        base_loss = effective_mark_loss(
            base.grain_optical_density,
            base.correlation_sigma_pixels,
            base.truncate,
        )
        compiled_loss = effective_mark_loss(
            candidate.grain_optical_density,
            candidate.correlation_sigma_pixels,
            candidate.truncate,
        )
        output.append(
            DensityConditionedLayerProfile(
                layer_id=candidate.layer_id,
                grain_optical_density=candidate.grain_optical_density,
                correlation_sigma_pixels=(
                    candidate.correlation_sigma_pixels
                ),
                seed=candidate.seed,
                count_rate_density=(
                    base.grain_optical_density
                    * compiled_loss
                    / base_loss
                ),
                boundary_mode=candidate.boundary_mode,
                maximum_target_density=candidate.maximum_target_density,
                truncate=candidate.truncate,
            )
        )
    return tuple(output)


def compile_two_cumulant_profiles(
    profiles: tuple[DensityConditionedLayerProfile, ...],
    *,
    pixel_size_factor: int,
    seed_offset: int = 0,
) -> tuple[DensityConditionedLayerProfile, ...]:
    """Match analytic area mean and shot-noise variance at a coarser LOD."""

    if not isinstance(pixel_size_factor, int) or pixel_size_factor < 1:
        raise ValueError("pixel_size_factor must be a positive integer")
    if not isinstance(seed_offset, int):
        raise ValueError("seed_offset must be an integer")
    output = []
    box = np.full(
        (pixel_size_factor, pixel_size_factor),
        1.0 / float(pixel_size_factor * pixel_size_factor),
        dtype=np.float64,
    )
    for base in profiles:
        base_kernel = _discrete_gaussian_kernel_2d(
            base.correlation_sigma_pixels, base.truncate
        )
        area_kernel = convolve2d(base_kernel, box, mode="full")
        reference_variance_coefficient = (
            base.grain_optical_density
            * float(np.sum(area_kernel * area_kernel, dtype=np.float64))
        )
        compiled_sigma = (
            base.correlation_sigma_pixels / pixel_size_factor
        )
        compiled_kernel = _discrete_gaussian_kernel_2d(
            compiled_sigma, base.truncate
        )
        compiled_kernel_square_sum = float(
            np.sum(compiled_kernel * compiled_kernel, dtype=np.float64)
        )
        base_loss_per_density = (
            effective_mark_loss(
                base.grain_optical_density,
                base.correlation_sigma_pixels,
                base.truncate,
            )
            / base.grain_optical_density
        )

        def residual(grain_density: float) -> float:
            compiled_loss = effective_mark_loss(
                grain_density, compiled_sigma, base.truncate
            )
            return (
                grain_density
                * grain_density
                * compiled_kernel_square_sum
                * base_loss_per_density
                / compiled_loss
                - reference_variance_coefficient
            )

        lower = 0.000001
        upper = base.grain_optical_density
        if residual(lower) >= 0.0 or residual(upper) <= 0.0:
            raise RuntimeError("two-cumulant root is not bracketed")
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            if residual(midpoint) > 0.0:
                upper = midpoint
            else:
                lower = midpoint
        compiled_grain_density = 0.5 * (lower + upper)
        compiled_loss = effective_mark_loss(
            compiled_grain_density, compiled_sigma, base.truncate
        )
        output.append(
            DensityConditionedLayerProfile(
                layer_id=base.layer_id,
                grain_optical_density=compiled_grain_density,
                correlation_sigma_pixels=compiled_sigma,
                seed=(base.seed + seed_offset) % (2**64),
                count_rate_density=(
                    compiled_loss / base_loss_per_density
                ),
                boundary_mode=base.boundary_mode,
                maximum_target_density=base.maximum_target_density,
                truncate=base.truncate,
            )
        )
    return tuple(output)


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
    rate = density_region / profile.resolved_count_rate_density
    counts = counter_poisson_rate_field(
        rate,
        full_shape,
        origin_yx=(y0, x0),
        seed=profile.seed,
        maximum_rate=(
            profile.maximum_target_density
            / profile.resolved_count_rate_density
        ),
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
        if profile.boundary_mode == "normalized-support-v1":
            support = gaussian_filter(
                np.ones(counts.shape, dtype=np.float64),
                sigma=sigma,
                order=0,
                mode="constant",
                cval=0.0,
                truncate=profile.truncate,
            )
            if np.any(support <= 0.0):
                raise RuntimeError("normalized Gaussian support is not positive")
            counts /= support
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


def render_density_conditioned_structure_area_lod_region(
    target_density: np.ndarray,
    profiles: tuple[DensityConditionedLayerProfile, ...],
    *,
    pixel_size_factor: int,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> DensityConditionedStructureResult:
    """Area-resolve physical subpixels and average film transmittance."""

    target = np.asarray(target_density, dtype=np.float64)
    if (
        target.ndim != 3
        or not isinstance(pixel_size_factor, int)
        or pixel_size_factor < 1
    ):
        raise ValueError("invalid area-LOD target or pixel factor")
    factor = pixel_size_factor
    expanded = np.repeat(
        np.repeat(target, factor, axis=0), factor, axis=1
    )
    origin_y, origin_x = origin_yx
    height, width = shape
    high = render_density_conditioned_structure_region(
        expanded,
        profiles,
        origin_yx=(origin_y * factor, origin_x * factor),
        shape=(height * factor, width * factor),
    )
    transmittance = high.transmittance.reshape(
        height,
        factor,
        width,
        factor,
        len(profiles),
    ).mean(axis=(1, 3), dtype=np.float64)
    density = -np.log(transmittance)
    return DensityConditionedStructureResult(
        density=density.astype(np.float32),
        transmittance=transmittance.astype(np.float32),
    )


def render_density_conditioned_structure_area_lod(
    target_density: np.ndarray,
    profiles: tuple[DensityConditionedLayerProfile, ...],
    *,
    pixel_size_factor: int,
) -> DensityConditionedStructureResult:
    target = np.asarray(target_density)
    if target.ndim != 3:
        raise ValueError("target density must be HxWxC")
    return render_density_conditioned_structure_area_lod_region(
        target,
        profiles,
        pixel_size_factor=pixel_size_factor,
        origin_yx=(0, 0),
        shape=target.shape[:2],
    )


def adaptive_exact_area_mask(
    target_density: np.ndarray,
    profiles: tuple[DensityConditionedLayerProfile, ...],
    *,
    pixel_size_factor: int,
    density_range_threshold: float,
) -> np.ndarray:
    """Select nonstationary pixels for exact base-pitch area execution."""

    target = np.asarray(target_density, dtype=np.float64)
    if (
        target.ndim != 3
        or target.shape[2] != len(profiles)
        or not isinstance(pixel_size_factor, int)
        or pixel_size_factor < 1
        or not math.isfinite(density_range_threshold)
        or density_range_threshold <= 0.0
    ):
        raise ValueError("invalid adaptive exact-area mask inputs")
    local_range = maximum_filter(
        target, size=(3, 3, 1), mode="nearest"
    ) - minimum_filter(target, size=(3, 3, 1), mode="nearest")
    mask = np.max(local_range, axis=2) > density_range_threshold
    maximum_radius = max(
        int(
            profile.truncate
            * profile.correlation_sigma_pixels
            / pixel_size_factor
            + 0.5
        )
        for profile in profiles
    )
    mask = binary_dilation(mask, iterations=maximum_radius + 1)
    output = np.asarray(mask, dtype=bool)
    output.setflags(write=False)
    return output


def render_density_conditioned_structure_adaptive_lod_region(
    target_density: np.ndarray,
    profiles: tuple[DensityConditionedLayerProfile, ...],
    *,
    pixel_size_factor: int,
    density_range_threshold: float,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> DensityConditionedStructureResult:
    """Use two-cumulant LOD in smooth pixels and exact area LOD elsewhere."""

    target = np.asarray(target_density, dtype=np.float64)
    mask = adaptive_exact_area_mask(
        target,
        profiles,
        pixel_size_factor=pixel_size_factor,
        density_range_threshold=density_range_threshold,
    )
    compiled = compile_two_cumulant_profiles(
        profiles, pixel_size_factor=pixel_size_factor
    )
    smooth = render_density_conditioned_structure_region(
        target,
        compiled,
        origin_yx=origin_yx,
        shape=shape,
    )
    exact = render_density_conditioned_structure_area_lod_region(
        target,
        profiles,
        pixel_size_factor=pixel_size_factor,
        origin_yx=origin_yx,
        shape=shape,
    )
    y0, x0 = origin_yx
    height, width = shape
    selection = mask[y0 : y0 + height, x0 : x0 + width, None]
    density = np.where(selection, exact.density, smooth.density)
    transmittance = np.where(
        selection, exact.transmittance, smooth.transmittance
    )
    return DensityConditionedStructureResult(
        density=np.asarray(density, dtype=np.float32),
        transmittance=np.asarray(transmittance, dtype=np.float32),
    )


def render_density_conditioned_structure_adaptive_lod(
    target_density: np.ndarray,
    profiles: tuple[DensityConditionedLayerProfile, ...],
    *,
    pixel_size_factor: int,
    density_range_threshold: float,
) -> DensityConditionedStructureResult:
    target = np.asarray(target_density)
    if target.ndim != 3:
        raise ValueError("target density must be HxWxC")
    return render_density_conditioned_structure_adaptive_lod_region(
        target,
        profiles,
        pixel_size_factor=pixel_size_factor,
        density_range_threshold=density_range_threshold,
        origin_yx=(0, 0),
        shape=target.shape[:2],
    )


__all__ = [
    "DENSITY_CONDITIONED_STRUCTURE_SCHEMA",
    "DensityConditionedLayerProfile",
    "DensityConditionedStructureResult",
    "adaptive_exact_area_mask",
    "compile_density_conditioned_profiles",
    "compile_effective_mark_loss_profiles",
    "compile_two_cumulant_profiles",
    "counter_poisson_rate_field",
    "effective_mark_loss",
    "render_density_conditioned_structure",
    "render_density_conditioned_structure_adaptive_lod",
    "render_density_conditioned_structure_adaptive_lod_region",
    "render_density_conditioned_structure_area_lod",
    "render_density_conditioned_structure_area_lod_region",
    "render_density_conditioned_structure_region",
]
