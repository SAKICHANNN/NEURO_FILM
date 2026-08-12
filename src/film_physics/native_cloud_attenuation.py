"""ctypes bridge for the P4DZ native cloud attenuation kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


def load_native_cloud_attenuation(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path.resolve()))
    pointer = ctypes.POINTER(ctypes.c_float)
    library.nf_cloud_attenuation_f32_apply_v1.argtypes = [
        pointer,
        pointer,
        ctypes.c_size_t,
        pointer,
        pointer,
        pointer,
    ]
    library.nf_cloud_attenuation_f32_apply_v1.restype = ctypes.c_int
    return library


def apply_native_cloud_attenuation(
    library: ctypes.CDLL,
    expected_transmittance_rgb: np.ndarray,
    base_transmittance_rgb: np.ndarray,
    channel_gain_rgb: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    expected = np.ascontiguousarray(expected_transmittance_rgb, dtype=np.float32)
    base = np.ascontiguousarray(base_transmittance_rgb, dtype=np.float32)
    gain = np.ascontiguousarray(channel_gain_rgb, dtype=np.float32)
    if expected.shape != base.shape or expected.ndim != 3 or expected.shape[-1] != 3:
        raise ValueError("native cloud attenuation requires matching HxWx3 arrays")
    density = np.empty_like(expected)
    output = np.empty_like(expected)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = library.nf_cloud_attenuation_f32_apply_v1(
        expected.ctypes.data_as(pointer),
        base.ctypes.data_as(pointer),
        expected.size // 3,
        gain.ctypes.data_as(pointer),
        density.ctypes.data_as(pointer),
        output.ctypes.data_as(pointer),
    )
    if status != 0:
        raise ValueError(f"native cloud attenuation rejected request: {status}")
    return density, output


__all__ = ["apply_native_cloud_attenuation", "load_native_cloud_attenuation"]
