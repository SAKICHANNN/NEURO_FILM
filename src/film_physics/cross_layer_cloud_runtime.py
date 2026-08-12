"""Bounded row-stream executor for a compiled cross-layer cloud profile."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter

from .cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from .cross_layer_compound_poisson import (
    sample_cross_layer_poisson_region,
    sample_density_conditioned_cross_layer_poisson_region,
)
from .density_conditioned_structure import DensityConditionedStructureResult


def _validated_runtime_profile(profile: CrossLayerCloudReferenceProfile, seed: int):
    if profile.product_enabled or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("invalid cross-layer cloud runtime request")
    return replace(profile.count_profile, seed=seed)


def _sample_periodic_row_halo(
    profile,
    full_shape: tuple[int, int],
    *,
    start_y: int,
    row_count: int,
) -> np.ndarray:
    height, width = full_shape
    if height < 1 or width < 1 or row_count < 1 or row_count > height:
        raise ValueError("invalid periodic row halo request")
    pieces = []
    remaining = row_count
    logical_y = start_y % height
    while remaining:
        take = min(remaining, height - logical_y)
        pieces.append(
            sample_cross_layer_poisson_region(
                profile,
                full_shape,
                origin_yx=(logical_y, 0),
                shape=(take, width),
            )
        )
        remaining -= take
        logical_y = 0
    output = np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
    output.setflags(write=False)
    return output


def iter_cross_layer_cloud_profile_rows(
    profile: CrossLayerCloudReferenceProfile,
    full_shape: tuple[int, int],
    *,
    seed: int,
    row_tile_height: int,
) -> Iterator[tuple[int, DensityConditionedStructureResult]]:
    """Yield exact full-width density/transmittance rows in raster order."""

    if (
        len(full_shape) != 2
        or any(not isinstance(value, int) or value < 1 for value in full_shape)
        or not isinstance(row_tile_height, int)
        or row_tile_height < 1
    ):
        raise ValueError("invalid cross-layer cloud row-stream geometry")
    count_profile = _validated_runtime_profile(profile, seed)
    halo = max(
        int(profile.gaussian_truncate * sigma + 0.5)
        for sigma in profile.gaussian_sigma_pixels_cmy
    )
    if full_shape[0] <= 2 * halo:
        raise ValueError("row-stream height must exceed twice the Gaussian halo")
    for y0 in range(0, full_shape[0], row_tile_height):
        rows = min(row_tile_height, full_shape[0] - y0)
        counts = _sample_periodic_row_halo(
            count_profile,
            full_shape,
            start_y=y0 - halo,
            row_count=rows + 2 * halo,
        )
        layers = []
        for channel, sigma in enumerate(profile.gaussian_sigma_pixels_cmy):
            filtered = gaussian_filter(
                counts[..., channel].astype(np.float64),
                sigma=sigma,
                mode=("nearest", "wrap"),
                truncate=profile.gaussian_truncate,
            )
            layers.append(
                filtered[halo : halo + rows]
                * profile.count_profile.mark_optical_density_cmy[channel]
            )
        density = np.stack(layers, axis=-1).astype(np.float32)
        transmittance = np.exp(-density.astype(np.float64)).astype(np.float32)
        yield y0, DensityConditionedStructureResult(density, transmittance)


def iter_density_conditioned_cross_layer_cloud_rows(
    profile: CrossLayerCloudReferenceProfile,
    scale_cmy: np.ndarray,
    *,
    seed: int,
    row_tile_height: int,
) -> Iterator[tuple[int, DensityConditionedStructureResult]]:
    """Yield exact rows for a spatial developed-density scale field."""

    scale = np.asarray(scale_cmy, dtype=np.float64)
    if (
        scale.ndim != 3
        or scale.shape[-1] != 3
        or not np.all(np.isfinite(scale))
        or np.any(scale < 0.0)
        or np.any(scale > 1.0)
        or not isinstance(row_tile_height, int)
        or row_tile_height < 1
    ):
        raise ValueError("invalid conditioned cross-layer cloud request")
    count_profile = _validated_runtime_profile(profile, seed)
    height, width = scale.shape[:2]
    halo = max(
        int(profile.gaussian_truncate * sigma + 0.5)
        for sigma in profile.gaussian_sigma_pixels_cmy
    )
    if height <= 2 * halo:
        raise ValueError("conditioned field height must exceed twice the halo")
    for y0 in range(0, height, row_tile_height):
        rows = min(row_tile_height, height - y0)
        logical = np.arange(y0 - halo, y0 + rows + halo) % height
        counts_parts = []
        cursor = 0
        while cursor < logical.size:
            start = int(logical[cursor])
            run = min(logical.size - cursor, height - start)
            counts_parts.append(
                sample_density_conditioned_cross_layer_poisson_region(
                    count_profile,
                    scale[start : start + run],
                    (height, width),
                    origin_yx=(start, 0),
                )
            )
            cursor += run
        counts = (
            np.concatenate(counts_parts, axis=0)
            if len(counts_parts) > 1
            else counts_parts[0]
        )
        layers = []
        for channel, sigma in enumerate(profile.gaussian_sigma_pixels_cmy):
            filtered = gaussian_filter(
                counts[..., channel].astype(np.float64),
                sigma=sigma,
                mode=("nearest", "wrap"),
                truncate=profile.gaussian_truncate,
            )
            layers.append(
                filtered[halo : halo + rows]
                * profile.count_profile.mark_optical_density_cmy[channel]
            )
        density = np.stack(layers, axis=-1).astype(np.float32)
        transmittance = np.exp(-density.astype(np.float64)).astype(np.float32)
        yield y0, DensityConditionedStructureResult(density, transmittance)


def iter_target_density_cross_layer_cloud_rows(
    profile: CrossLayerCloudReferenceProfile,
    target_density_cmy: np.ndarray,
    *,
    maximum_developed_density_cmy: tuple[float, float, float],
    seed: int,
    row_tile_height: int,
) -> Iterator[tuple[int, DensityConditionedStructureResult]]:
    """Map typed developed optical density to bounded cloud occurrence rates."""

    target = np.asarray(target_density_cmy, dtype=np.float64)
    maximum = np.asarray(maximum_developed_density_cmy, dtype=np.float64)
    expected_maximum = np.asarray(
        profile.count_profile.marginal_rates_cmy
    ) * np.asarray(profile.count_profile.mark_optical_density_cmy)
    if (
        target.ndim != 3
        or target.shape[-1] != 3
        or maximum.shape != (3,)
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(maximum))
        or np.any(target < 0.0)
        or np.any(maximum <= 0.0)
        or not np.array_equal(maximum, expected_maximum)
        or np.any(target > maximum)
    ):
        raise ValueError(
            "target developed density is outside the compiled profile domain"
        )
    yield from iter_density_conditioned_cross_layer_cloud_rows(
        profile,
        target / maximum,
        seed=seed,
        row_tile_height=row_tile_height,
    )


def optical_density_capacity_cmy(
    profile: CrossLayerCloudReferenceProfile,
) -> tuple[float, float, float]:
    """Return exact homogeneous log10-density capacity of the cloud kernels."""

    capacities = []
    log_ten = math.log(10.0)
    for rate, mark, sigma in zip(
        profile.count_profile.marginal_rates_cmy,
        profile.count_profile.mark_optical_density_cmy,
        profile.gaussian_sigma_pixels_cmy,
        strict=True,
    ):
        radius = int(profile.gaussian_truncate * sigma + 0.5)
        impulse = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
        impulse[radius, radius] = 1.0
        kernel = gaussian_filter(
            impulse,
            sigma=sigma,
            mode="constant",
            cval=0.0,
            truncate=profile.gaussian_truncate,
        )
        log_expectation = rate * float(np.sum(np.expm1(-log_ten * mark * kernel)))
        capacities.append(-log_expectation / log_ten)
    return tuple(capacities)


def iter_optical_density_cross_layer_cloud_rows_v2(
    profile: CrossLayerCloudReferenceProfile,
    target_optical_density_cmy: np.ndarray,
    *,
    seed: int,
    row_tile_height: int,
) -> Iterator[tuple[int, DensityConditionedStructureResult]]:
    """Yield optical-density clouds with PGF-calibrated mean transmittance."""

    target = np.asarray(target_optical_density_cmy, dtype=np.float64)
    capacity = np.asarray(optical_density_capacity_cmy(profile), dtype=np.float64)
    if (
        target.ndim != 3
        or target.shape[-1] != 3
        or not np.all(np.isfinite(target))
        or np.any(target < 0.0)
        or np.any(target > capacity)
    ):
        raise ValueError("target optical density is outside the v2 cloud capacity")
    for y0, legacy in iter_density_conditioned_cross_layer_cloud_rows(
        profile,
        target / capacity,
        seed=seed,
        row_tile_height=row_tile_height,
    ):
        density = legacy.density
        transmittance = np.power(10.0, -density.astype(np.float64)).astype(np.float32)
        yield y0, DensityConditionedStructureResult(density, transmittance)


def estimate_cross_layer_cloud_row_stream_live_bytes(
    profile: CrossLayerCloudReferenceProfile,
    *,
    full_shape: tuple[int, int],
    row_tile_height: int,
) -> int:
    """Conservative live-array model for one yielded Python reference tile."""

    if (
        len(full_shape) != 2
        or any(not isinstance(value, int) or value < 1 for value in full_shape)
        or not isinstance(row_tile_height, int)
        or row_tile_height < 1
    ):
        raise ValueError("invalid row-stream workspace geometry")
    halo = max(
        int(profile.gaussian_truncate * sigma + 0.5)
        for sigma in profile.gaussian_sigma_pixels_cmy
    )
    rows = min(row_tile_height, full_shape[0])
    extended_pixels = (rows + 2 * halo) * full_shape[1]
    core_pixels = rows * full_shape[1]
    # uint16 three-channel counts + one float64 convolution workspace,
    # plus float32 density and transmittance retained for the yielded tile.
    return extended_pixels * (3 * 2 + 8) + core_pixels * (3 * 4 * 2)


def estimate_cross_layer_cloud_full_live_bytes(full_shape: tuple[int, int]) -> int:
    if len(full_shape) != 2 or any(
        not isinstance(value, int) or value < 1 for value in full_shape
    ):
        raise ValueError("invalid full-frame workspace geometry")
    return math.prod(full_shape) * (3 * 2 + 3 * 4 * 2)


__all__ = [
    "estimate_cross_layer_cloud_full_live_bytes",
    "estimate_cross_layer_cloud_row_stream_live_bytes",
    "iter_cross_layer_cloud_profile_rows",
    "iter_density_conditioned_cross_layer_cloud_rows",
    "iter_target_density_cross_layer_cloud_rows",
]
