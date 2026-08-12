"""Python bridge for the P4EI native conditioned cloud row chain."""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from .density_conditioned_structure import DensityConditionedStructureResult


class _CountProfile(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("marginal", ctypes.c_double * 3), ("shared_all", ctypes.c_double),
        ("pairs", ctypes.c_double * 3), ("seed", ctypes.c_uint64),
        ("stride", ctypes.c_uint64),
    ]


class _SpatialProfile(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("sigma", ctypes.c_double * 3), ("mark", ctypes.c_double * 3),
        ("truncate", ctypes.c_double),
    ]


def load_native_conditioned_cloud(count_library: Path, spatial_library: Path):
    count = ctypes.CDLL(str(count_library))
    spatial = ctypes.CDLL(str(spatial_library))
    count.nf_density_conditioned_poisson_u16_sample_region_v2.argtypes = [
        ctypes.POINTER(_CountProfile), *([ctypes.c_size_t] * 6),
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t,
    ]
    count.nf_density_conditioned_poisson_u16_sample_region_v2.restype = ctypes.c_int
    spatial.nf_cloud_spatial_response_f32_apply_v1.argtypes = [
        ctypes.POINTER(_SpatialProfile), ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ]
    spatial.nf_cloud_spatial_response_f32_apply_v1.restype = ctypes.c_int
    return count, spatial


def _sample_counts(library, profile, scale, full_shape, origin_y):
    values = np.ascontiguousarray(scale, dtype=np.float32)
    output = np.empty(values.shape, dtype=np.uint16)
    status = library.nf_density_conditioned_poisson_u16_sample_region_v2(
        ctypes.byref(profile), *full_shape, origin_y, 0, *values.shape[:2],
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), values.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), output.size,
    )
    if status != 0:
        raise ValueError(f"native conditioned count kernel rejected request: {status}")
    return output


def _spatial(library, profile, counts, core_height, halo):
    width = counts.shape[1]
    density = np.empty((core_height, width, 3), dtype=np.float32)
    transmittance = np.empty_like(density)
    workspace = np.empty(2 * counts.shape[0] * width, dtype=np.float64)
    status = library.nf_cloud_spatial_response_f32_apply_v1(
        ctypes.byref(profile), counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        core_height, width, halo,
        workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), workspace.size,
        density.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        transmittance.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), density.size,
    )
    if status != 0:
        raise ValueError(f"native cloud spatial kernel rejected request: {status}")
    return DensityConditionedStructureResult(density, transmittance)


def iter_native_density_conditioned_cloud_rows(
    profile: CrossLayerCloudReferenceProfile,
    scale_cmy: np.ndarray,
    *,
    seed: int,
    row_tile_height: int,
    count_library: object,
    spatial_library: object,
) -> Iterator[tuple[int, DensityConditionedStructureResult]]:
    scale = np.asarray(scale_cmy, dtype=np.float32)
    if (
        profile.product_enabled or scale.ndim != 3 or scale.shape[-1] != 3
        or not np.all(np.isfinite(scale)) or np.any(scale < 0.0) or np.any(scale > 1.0)
        or not isinstance(seed, int) or not 0 <= seed < 2**64
        or not isinstance(row_tile_height, int) or row_tile_height < 1
    ):
        raise ValueError("invalid native conditioned cloud row request")
    height, width = scale.shape[:2]
    halo = max(int(profile.gaussian_truncate * value + 0.5) for value in profile.gaussian_sigma_pixels_cmy)
    if height <= 2 * halo:
        raise ValueError("native conditioned field height must exceed twice the halo")
    count = profile.count_profile
    count_profile = _CountProfile(
        ctypes.sizeof(_CountProfile), 2, (ctypes.c_double * 3)(*count.marginal_rates_cmy),
        count.shared_all_rate, (ctypes.c_double * 3)(*count.shared_pair_rates_cm_cy_my),
        seed, count.component_seed_stride,
    )
    spatial_profile = _SpatialProfile(
        ctypes.sizeof(_SpatialProfile), 1,
        (ctypes.c_double * 3)(*profile.gaussian_sigma_pixels_cmy),
        (ctypes.c_double * 3)(*count.mark_optical_density_cmy),
        profile.gaussian_truncate,
    )
    for y0 in range(0, height, row_tile_height):
        rows = min(row_tile_height, height - y0)
        logical = np.arange(y0 - halo, y0 + rows + halo) % height
        parts = []
        cursor = 0
        while cursor < logical.size:
            start = int(logical[cursor])
            run = min(logical.size - cursor, height - start)
            parts.append(_sample_counts(count_library, count_profile, scale[start : start + run], (height, width), start))
            cursor += run
        counts = np.concatenate(parts) if len(parts) > 1 else parts[0]
        yield y0, _spatial(spatial_library, spatial_profile, counts, rows, halo)


__all__ = ["iter_native_density_conditioned_cloud_rows", "load_native_conditioned_cloud"]
