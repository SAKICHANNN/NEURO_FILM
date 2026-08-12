"""Bounded row-stream executor for a compiled cross-layer cloud profile."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter

from .cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from .cross_layer_compound_poisson import sample_cross_layer_poisson_region
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
]
