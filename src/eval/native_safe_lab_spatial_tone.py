"""Host-native feasibility boundary for safe-Lab spatial detail and tone."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll

SOURCE = "native/color_engine/nf_safe_lab_spatial_tone_f32_v1.c"
HEADER = "native/color_engine/nf_safe_lab_spatial_tone_f32_v1.h"


class NativeSafeLabSpatialToneError(RuntimeError):
    """Raised when the native spatial/tone kernel rejects an invocation."""


def build_native_safe_lab_spatial_tone(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative=SOURCE,
        header_relative=HEADER,
        basename="nf_safe_lab_spatial_tone_f32_v1",
    )


def load_native_safe_lab_spatial_tone(dll_path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(Path(dll_path).resolve()))
    function = library.nf_safe_lab_spatial_tone_f32_v1
    pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint32,
    ]
    function.restype = ctypes.c_int
    return library


def apply_native_safe_lab_spatial_tone(
    library: ctypes.CDLL,
    source_lab: np.ndarray,
    pointwise_lab: np.ndarray,
    *,
    sigma: float = 1.1,
    truncate: float = 4.0,
    detail_strength: float,
    tone_strength: float,
    shadow_floor_l: float,
    highlight_ceiling_l: float,
    thread_count: int = 8,
    output: np.ndarray | None = None,
    source_workspace: np.ndarray | None = None,
    target_workspace: np.ndarray | None = None,
) -> np.ndarray:
    if (
        not isinstance(source_lab, np.ndarray)
        or source_lab.dtype != np.float32
        or source_lab.ndim != 3
        or source_lab.shape[2] != 3
        or not source_lab.flags.c_contiguous
        or not isinstance(pointwise_lab, np.ndarray)
        or pointwise_lab.dtype != np.float32
        or pointwise_lab.shape != source_lab.shape
        or not pointwise_lab.flags.c_contiguous
    ):
        raise TypeError("source and pointwise Lab must be C-contiguous HxWx3 float32")
    shape = source_lab.shape
    plane_shape = shape[:2]
    if output is None:
        output = np.empty_like(source_lab)
    if source_workspace is None:
        source_workspace = np.empty(plane_shape, dtype=np.float32)
    if target_workspace is None:
        target_workspace = np.empty(plane_shape, dtype=np.float32)
    for array, expected in (
        (output, shape),
        (source_workspace, plane_shape),
        (target_workspace, plane_shape),
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.float32
            or array.shape != expected
            or not array.flags.c_contiguous
        ):
            raise TypeError("native spatial/tone buffer has invalid shape or layout")
    pointer = ctypes.POINTER(ctypes.c_float)
    status = library.nf_safe_lab_spatial_tone_f32_v1(
        source_lab.ctypes.data_as(pointer),
        pointwise_lab.ctypes.data_as(pointer),
        output.ctypes.data_as(pointer),
        source_workspace.ctypes.data_as(pointer),
        target_workspace.ctypes.data_as(pointer),
        shape[0],
        shape[1],
        float(sigma),
        float(truncate),
        float(detail_strength),
        float(tone_strength),
        float(shadow_floor_l),
        float(highlight_ceiling_l),
        int(thread_count),
    )
    if status != 0:
        raise NativeSafeLabSpatialToneError(
            f"native spatial/tone safe-Lab rejected status {status}"
        )
    return output


__all__ = [
    "NativeSafeLabSpatialToneError",
    "apply_native_safe_lab_spatial_tone",
    "build_native_safe_lab_spatial_tone",
    "load_native_safe_lab_spatial_tone",
]
