"""Host binding for the portable P4BS/P4BV Thomas-field C ABI."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class NativeThomasFieldError(RuntimeError):
    """Raised when the native Thomas ABI rejects a request."""


class NativeThomasFieldProfileV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("particle_sigma_pixels", ctypes.c_double),
        ("cluster_sigma_pixels", ctypes.c_double),
        ("mean_offspring", ctypes.c_double),
        ("truncate", ctypes.c_double),
        ("component_seeds", ctypes.c_uint64 * 2),
        ("realization_seed", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class NativeThomasFieldProfile:
    particle_sigma_pixels: float
    cluster_sigma_pixels: float
    mean_offspring: float
    truncate: float
    component_seeds: tuple[int, int]
    realization_seed: int

    def as_abi(self) -> NativeThomasFieldProfileV1:
        return NativeThomasFieldProfileV1(
            ctypes.sizeof(NativeThomasFieldProfileV1),
            1,
            self.particle_sigma_pixels,
            self.cluster_sigma_pixels,
            self.mean_offspring,
            self.truncate,
            (ctypes.c_uint64 * 2)(*self.component_seeds),
            self.realization_seed,
        )


def _float_pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def load_native_thomas_field_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path.resolve()))
    library.nf_thomas_field_f32_abi_version_v1.argtypes = []
    library.nf_thomas_field_f32_abi_version_v1.restype = ctypes.c_uint32
    library.nf_thomas_field_f32_workspace_floats_v1.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.nf_thomas_field_f32_workspace_floats_v1.restype = ctypes.c_int
    library.nf_thomas_field_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeThomasFieldProfileV1),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_thomas_field_f32_apply_v1.restype = ctypes.c_int
    if library.nf_thomas_field_f32_abi_version_v1() != 1:
        raise NativeThomasFieldError("native Thomas ABI version mismatch")
    return library


def render_native_thomas_field(
    library: ctypes.CDLL,
    profile: NativeThomasFieldProfile,
    shape: tuple[int, int],
) -> tuple[np.ndarray, float, int]:
    height, width = shape
    abi_profile = profile.as_abi()
    workspace_count = ctypes.c_size_t()
    status = library.nf_thomas_field_f32_workspace_floats_v1(
        height, width, ctypes.byref(workspace_count)
    )
    if status != 0:
        raise NativeThomasFieldError(f"native workspace request failed: {status}")
    workspace = np.empty(workspace_count.value, dtype=np.float32)
    output = np.empty(height * width, dtype=np.float32)
    raw_mean = ctypes.c_double()
    status = library.nf_thomas_field_f32_apply_v1(
        ctypes.byref(abi_profile),
        height,
        width,
        _float_pointer(workspace),
        workspace.size,
        _float_pointer(output),
        output.size,
        ctypes.byref(raw_mean),
    )
    if status != 0:
        raise NativeThomasFieldError(f"native Thomas render failed: {status}")
    return output.reshape(shape), raw_mean.value, workspace.nbytes


__all__ = [
    "NativeThomasFieldError",
    "NativeThomasFieldProfile",
    "NativeThomasFieldProfileV1",
    "load_native_thomas_field_library",
    "render_native_thomas_field",
]
