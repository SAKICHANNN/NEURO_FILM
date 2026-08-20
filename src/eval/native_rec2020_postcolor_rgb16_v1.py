"""Host-native exact residual, BT.2020 OETF, and RGB16 quantization kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll
from src.preprocess import linear_rec2020_to_rec2020

SOURCE = "native/color_engine/nf_rec2020_postcolor_rgb16_f32_v1.c"
HEADER = "native/color_engine/nf_rec2020_postcolor_rgb16_f32_v1.h"


def build_exact_rec2020_rgb16_thresholds() -> np.ndarray:
    """Return the first float32 input that quantizes above each RGB16 code."""

    codes = np.arange(65535, dtype=np.uint16)
    low = np.zeros(65535, dtype=np.uint32)
    high = np.full(65535, np.float32(1.0).view(np.uint32), dtype=np.uint32)
    while bool(np.any(low < high)):
        mid = low + (high - low) // np.uint32(2)
        values = np.repeat(mid.view(np.float32).reshape(-1, 1, 1), 3, axis=2)
        encoded = linear_rec2020_to_rec2020(values)[:, 0, 0]
        quantized = np.rint(encoded * 65535.0).astype(np.uint16)
        above = quantized > codes
        high = np.where(above, mid, high)
        low = np.where(above, low, mid + np.uint32(1))
    return np.ascontiguousarray(low.view(np.float32))


def build_exact_rec2020_rgb16_buckets(thresholds: np.ndarray) -> np.ndarray:
    """Bound exact threshold searches by the high 16 float bits."""

    threshold_bits = thresholds.view(np.uint32)
    boundaries = np.minimum(
        np.arange(65537, dtype=np.uint64) << np.uint64(16),
        np.uint64(np.iinfo(np.uint32).max),
    )
    return np.ascontiguousarray(
        np.searchsorted(threshold_bits, boundaries, side="left").astype(np.uint32)
    )


class NativeRec2020PostcolorError(RuntimeError):
    """Raised when the native post-colour kernel rejects an invocation."""


def build_native_rec2020_postcolor_v1(*, root: Path, output_dir: Path) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative=SOURCE,
        header_relative=HEADER,
        basename="nf_rec2020_postcolor_rgb16_f32_v1",
    )


def load_native_rec2020_postcolor_v1(dll_path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(Path(dll_path).resolve()))
    function = library.nf_rec2020_postcolor_rgb16_f32_v1
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.c_uint32,
    ]
    function.restype = ctypes.c_int
    return library


def apply_native_rec2020_postcolor_v1(
    library: ctypes.CDLL,
    source: np.ndarray,
    candidate: np.ndarray,
    margin: float,
    *,
    thread_count: int = 8,
    thresholds: np.ndarray | None = None,
    buckets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        not isinstance(source, np.ndarray)
        or not isinstance(candidate, np.ndarray)
        or source.dtype != np.float32
        or candidate.dtype != np.float32
        or source.ndim != 3
        or source.shape[2] != 3
        or candidate.shape != source.shape
        or not source.flags.c_contiguous
        or not candidate.flags.c_contiguous
    ):
        raise TypeError("source and candidate must be matching C-contiguous HxWx3 float32")
    output = np.empty(source.shape, dtype=np.uint16)
    scale = np.empty(source.shape[:2], dtype=np.float32)
    thresholds = (
        build_exact_rec2020_rgb16_thresholds() if thresholds is None else thresholds
    )
    if (
        thresholds.dtype != np.float32
        or thresholds.shape != (65535,)
        or not thresholds.flags.c_contiguous
    ):
        raise TypeError("thresholds must be C-contiguous float32[65535]")
    buckets = build_exact_rec2020_rgb16_buckets(thresholds) if buckets is None else buckets
    if (
        buckets.dtype != np.uint32
        or buckets.shape != (65537,)
        or not buckets.flags.c_contiguous
    ):
        raise TypeError("buckets must be C-contiguous uint32[65537]")
    status = library.nf_rec2020_postcolor_rgb16_f32_v1(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        candidate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        thresholds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        buckets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        source.shape[0] * source.shape[1],
        thresholds.size,
        buckets.size,
        float(margin),
        int(thread_count),
    )
    if status != 0:
        raise NativeRec2020PostcolorError(
            f"native Rec.2020 post-colour kernel rejected status {status}"
        )
    return output, scale


__all__ = [
    "NativeRec2020PostcolorError",
    "apply_native_rec2020_postcolor_v1",
    "build_exact_rec2020_rgb16_buckets",
    "build_exact_rec2020_rgb16_thresholds",
    "build_native_rec2020_postcolor_v1",
    "load_native_rec2020_postcolor_v1",
]
