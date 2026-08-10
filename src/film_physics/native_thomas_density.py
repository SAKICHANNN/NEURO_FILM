"""Host binding for native developed-density Thomas composition."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from .native_thomas_field import NativeThomasFieldProfile, NativeThomasFieldProfileV1


class NativeThomasDensityError(RuntimeError):
    """Raised when the native density-domain ABI rejects a request."""


def _pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def load_native_thomas_density_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path.resolve()))
    library.nf_thomas_density_f32_abi_version_v1.argtypes = []
    library.nf_thomas_density_f32_abi_version_v1.restype = ctypes.c_uint32
    library.nf_thomas_density_f32_workspace_floats_v1.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.nf_thomas_density_f32_workspace_floats_v1.restype = ctypes.c_int
    library.nf_thomas_density_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeThomasFieldProfileV1),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_thomas_density_f32_apply_v1.restype = ctypes.c_int
    if library.nf_thomas_density_f32_abi_version_v1() != 1:
        raise NativeThomasDensityError("native Thomas density ABI version mismatch")
    return library


def render_native_thomas_transmittance(
    library: ctypes.CDLL,
    profile: NativeThomasFieldProfile,
    base_density: np.ndarray,
    point_density_sigma: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    base = np.ascontiguousarray(base_density, dtype=np.float32)
    sigma = np.ascontiguousarray(point_density_sigma, dtype=np.float32)
    if base.ndim != 2 or sigma.shape != base.shape:
        raise ValueError("density and sigma must be matching 2-D arrays")
    height, width = base.shape
    workspace_count = ctypes.c_size_t()
    status = library.nf_thomas_density_f32_workspace_floats_v1(
        height, width, ctypes.byref(workspace_count)
    )
    if status != 0:
        raise NativeThomasDensityError(f"native workspace request failed: {status}")
    workspace = np.empty(workspace_count.value, dtype=np.float32)
    output = np.empty(base.size, dtype=np.float32)
    raw_mean = ctypes.c_double()
    abi_profile = profile.as_abi()
    status = library.nf_thomas_density_f32_apply_v1(
        ctypes.byref(abi_profile),
        height,
        width,
        _pointer(base),
        base.size,
        _pointer(sigma),
        sigma.size,
        _pointer(workspace),
        workspace.size,
        _pointer(output),
        output.size,
        ctypes.byref(raw_mean),
    )
    if status != 0:
        raise NativeThomasDensityError(f"native density composition failed: {status}")
    return output.reshape(base.shape), raw_mean.value, workspace.nbytes


def render_native_thomas_rgb_transmittance(
    library: ctypes.CDLL,
    profiles: tuple[
        NativeThomasFieldProfile,
        NativeThomasFieldProfile,
        NativeThomasFieldProfile,
    ],
    base_density_chw: np.ndarray,
    point_density_sigma_chw: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float], int]:
    """Render three independent planar layers with one reused workspace."""

    base = np.asarray(base_density_chw)
    sigma = np.asarray(point_density_sigma_chw)
    if (
        base.dtype != np.float32
        or sigma.dtype != np.float32
        or base.ndim != 3
        or base.shape[0] != 3
        or sigma.shape != base.shape
        or not base.flags.c_contiguous
        or not sigma.flags.c_contiguous
    ):
        raise ValueError("RGB density and sigma must be contiguous float32 CHW arrays")
    _, height, width = base.shape
    count = height * width
    workspace_count = ctypes.c_size_t()
    status = library.nf_thomas_density_f32_workspace_floats_v1(
        height, width, ctypes.byref(workspace_count)
    )
    if status != 0:
        raise NativeThomasDensityError(f"native workspace request failed: {status}")
    workspace = np.empty(workspace_count.value, dtype=np.float32)
    scratch = np.empty(count, dtype=np.float32)
    output = np.empty_like(base)
    raw_means: list[float] = []
    for channel, profile in enumerate(profiles):
        raw_mean = ctypes.c_double()
        abi_profile = profile.as_abi()
        status = library.nf_thomas_density_f32_apply_v1(
            ctypes.byref(abi_profile),
            height,
            width,
            _pointer(base[channel]),
            count,
            _pointer(sigma[channel]),
            count,
            _pointer(workspace),
            workspace.size,
            _pointer(scratch),
            scratch.size,
            ctypes.byref(raw_mean),
        )
        if status != 0:
            raise NativeThomasDensityError(
                f"native RGB density composition failed at channel {channel}: {status}"
            )
        output[channel] = scratch.reshape(height, width)
        raw_means.append(raw_mean.value)
    output.setflags(write=False)
    return output, tuple(raw_means), workspace.nbytes + scratch.nbytes  # type: ignore[return-value]


__all__ = [
    "NativeThomasDensityError",
    "load_native_thomas_density_library",
    "render_native_thomas_rgb_transmittance",
    "render_native_thomas_transmittance",
]
