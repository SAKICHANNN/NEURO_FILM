"""Host-native feasibility boundary for Rec.2020 Lab source compression."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll

SOURCE = "native/color_engine/nf_rec2020_lab_source_compress_f32_v1.c"
HEADER = "native/color_engine/nf_rec2020_lab_source_compress_f32_v1.h"


class NativeRec2020LabCompressError(RuntimeError):
    """Raised when the native Lab compression kernel rejects an invocation."""


def build_native_rec2020_lab_compress(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative=SOURCE,
        header_relative=HEADER,
        basename="nf_rec2020_lab_source_compress_f32_v1",
    )


def load_native_rec2020_lab_compress(dll_path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(Path(dll_path).resolve()))
    function = library.nf_rec2020_lab_source_compress_f32_v1
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.c_uint32,
    ]
    function.restype = ctypes.c_int
    return library


def apply_native_rec2020_lab_compress(
    library: ctypes.CDLL,
    source_lab: np.ndarray,
    target_lab: np.ndarray,
    *,
    iterations: int = 24,
    tolerance: float = 2e-6,
    thread_count: int = 8,
    output_lab: np.ndarray | None = None,
    output_rgb: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        not isinstance(source_lab, np.ndarray)
        or not isinstance(target_lab, np.ndarray)
        or source_lab.dtype != np.float32
        or target_lab.dtype != np.float32
        or source_lab.ndim != 3
        or source_lab.shape[2] != 3
        or target_lab.shape != source_lab.shape
        or not source_lab.flags.c_contiguous
        or not target_lab.flags.c_contiguous
    ):
        raise TypeError("source_lab and target_lab must be matching C-contiguous HxWx3 float32")
    if output_lab is None:
        output_lab = np.empty_like(source_lab)
    if output_rgb is None:
        output_rgb = np.empty_like(source_lab)
    if scale is None:
        scale = np.empty(source_lab.shape[:2], dtype=np.float32)
    if (
        output_lab.dtype != np.float32
        or output_lab.shape != source_lab.shape
        or not output_lab.flags.c_contiguous
        or output_rgb.dtype != np.float32
        or output_rgb.shape != source_lab.shape
        or not output_rgb.flags.c_contiguous
        or scale.dtype != np.float32
        or scale.shape != source_lab.shape[:2]
        or not scale.flags.c_contiguous
    ):
        raise TypeError("native output buffers have invalid shape, dtype, or layout")
    status = library.nf_rec2020_lab_source_compress_f32_v1(
        source_lab.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        target_lab.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output_lab.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        source_lab.shape[0] * source_lab.shape[1],
        int(iterations),
        float(tolerance),
        int(thread_count),
    )
    if status != 0:
        raise NativeRec2020LabCompressError(
            f"native Rec.2020 Lab compression rejected status {status}"
        )
    return output_lab, output_rgb, scale


__all__ = [
    "NativeRec2020LabCompressError",
    "apply_native_rec2020_lab_compress",
    "build_native_rec2020_lab_compress",
    "load_native_rec2020_lab_compress",
]
