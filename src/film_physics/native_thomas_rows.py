"""Bounded-row host binding for the unchanged native Thomas field."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from .native_granularity_amplitude import (
    NativeGranularityAmplitudeError,
    NativeGranularityAmplitudeProfileV1,
)
from .native_thomas_field import NativeThomasFieldProfile, NativeThomasFieldProfileV1


class NativeThomasRowsError(RuntimeError):
    """Raised when the native global-coordinate row ABI rejects a request."""


def _pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def load_native_thomas_rows_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path.resolve()))
    library.nf_thomas_rows_f32_abi_version_v1.argtypes = []
    library.nf_thomas_rows_f32_abi_version_v1.restype = ctypes.c_uint32
    library.nf_thomas_rows_f32_workspace_floats_v1.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.nf_thomas_rows_f32_workspace_floats_v1.restype = ctypes.c_int
    library.nf_thomas_rows_f32_field_v1.argtypes = [
        ctypes.POINTER(NativeThomasFieldProfileV1),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    library.nf_thomas_rows_f32_field_v1.restype = ctypes.c_int
    library.nf_thomas_rows_f32_density_v1.argtypes = [
        ctypes.POINTER(NativeThomasFieldProfileV1),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    library.nf_thomas_rows_f32_density_v1.restype = ctypes.c_int
    if library.nf_thomas_rows_f32_abi_version_v1() != 1:
        raise NativeThomasRowsError("native Thomas rows ABI version mismatch")
    return library


def _workspace(library: ctypes.CDLL, width: int, rows: int) -> np.ndarray:
    count = ctypes.c_size_t()
    status = library.nf_thomas_rows_f32_workspace_floats_v1(
        width, rows, ctypes.byref(count)
    )
    if status != 0:
        raise NativeThomasRowsError(f"native row workspace request failed: {status}")
    return np.empty(count.value, dtype=np.float32)


def _neumaier_rows(
    library: ctypes.CDLL,
    profile: NativeThomasFieldProfile,
    shape: tuple[int, int],
    row_partition: int,
) -> float:
    height, width = shape
    workspace = _workspace(library, width, min(height, row_partition))
    output = np.empty(min(height, row_partition) * width, dtype=np.float32)
    abi_profile = profile.as_abi()
    total = 0.0
    compensation = 0.0
    for row_start in range(0, height, row_partition):
        rows = min(row_partition, height - row_start)
        status = library.nf_thomas_rows_f32_field_v1(
            ctypes.byref(abi_profile),
            height,
            width,
            row_start,
            rows,
            _pointer(workspace),
            workspace.size,
            _pointer(output),
            output.size,
        )
        if status != 0:
            raise NativeThomasRowsError(
                f"native field-row pass failed at row {row_start}: {status}"
            )
        for value in output[: rows * width]:
            sample = float(value)
            updated = total + sample
            if abs(total) >= abs(sample):
                compensation += (total - updated) + sample
            else:
                compensation += (sample - updated) + total
            total = updated
    return (total + compensation) / float(height * width)


def render_native_exposure_thomas_rgb_rows(
    amplitude_library: ctypes.CDLL,
    rows_library: ctypes.CDLL,
    amplitude_profile: NativeGranularityAmplitudeProfileV1,
    field_profiles: tuple[
        NativeThomasFieldProfile,
        NativeThomasFieldProfile,
        NativeThomasFieldProfile,
    ],
    relative_log_exposure_chw: np.ndarray,
    *,
    row_partition: int,
) -> tuple[np.ndarray, tuple[float, float, float], int]:
    """Two-pass exact P8BW execution with bounded row-local intermediates."""

    exposure = np.asarray(relative_log_exposure_chw)
    if (
        exposure.dtype != np.float32
        or exposure.ndim != 3
        or exposure.shape[0] != 3
        or not exposure.flags.c_contiguous
        or isinstance(row_partition, bool)
        or not isinstance(row_partition, int)
        or row_partition <= 0
    ):
        raise ValueError("expected contiguous float32 CHW exposure and positive rows")
    for channel in range(3):
        values = exposure[channel]
        count = int(amplitude_profile.knot_count[channel])
        lower = float(amplitude_profile.log_exposure_knots[channel][0])
        upper = float(amplitude_profile.log_exposure_knots[channel][count - 1])
        if (
            not np.all(np.isfinite(values))
            or float(np.min(values)) < lower
            or float(np.max(values)) > upper
        ):
            raise ValueError("relative layer log exposure is outside the profile domain")

    _, height, width = exposure.shape
    tile_rows = min(height, row_partition)
    workspace = _workspace(rows_library, width, tile_rows)
    density = np.empty(tile_rows * width, dtype=np.float32)
    sigma = np.empty(tile_rows * width, dtype=np.float32)
    tile_output = np.empty(tile_rows * width, dtype=np.float32)
    output = np.empty_like(exposure)
    raw_means = tuple(
        _neumaier_rows(rows_library, profile, (height, width), row_partition)
        for profile in field_profiles
    )
    for channel, field_profile in enumerate(field_profiles):
        abi_profile = field_profile.as_abi()
        for row_start in range(0, height, row_partition):
            rows = min(row_partition, height - row_start)
            count = rows * width
            exposure_rows = exposure[channel, row_start : row_start + rows]
            status = amplitude_library.nf_granularity_amplitude_f32_apply_layer_v1(
                ctypes.byref(amplitude_profile),
                channel,
                _pointer(exposure_rows),
                count,
                _pointer(density),
                _pointer(sigma),
            )
            if status != 0:
                raise NativeGranularityAmplitudeError(
                    f"native row amplitude failed at channel {channel}: {status}"
                )
            status = rows_library.nf_thomas_rows_f32_density_v1(
                ctypes.byref(abi_profile),
                height,
                width,
                row_start,
                rows,
                _pointer(density),
                count,
                _pointer(sigma),
                count,
                raw_means[channel],
                _pointer(workspace),
                workspace.size,
                _pointer(tile_output),
                tile_output.size,
            )
            if status != 0:
                raise NativeThomasRowsError(
                    f"native row density pass failed at channel {channel}, "
                    f"row {row_start}: {status}"
                )
            output[channel, row_start : row_start + rows] = tile_output[
                :count
            ].reshape(rows, width)
    output.setflags(write=False)
    workspace_bytes = workspace.nbytes + density.nbytes + sigma.nbytes + tile_output.nbytes
    return output, raw_means, workspace_bytes


__all__ = [
    "NativeThomasRowsError",
    "load_native_thomas_rows_library",
    "render_native_exposure_thomas_rgb_rows",
]
