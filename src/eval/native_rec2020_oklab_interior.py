"""Host-native feasibility boundary for the C11 Rec.2020 OKLab mapper."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll

SOURCE = "native/color_engine/nf_rec2020_oklab_interior_f32_v1.c"
HEADER = "native/color_engine/nf_rec2020_oklab_interior_f32_v1.h"


class NativeRec2020InteriorError(RuntimeError):
    """Raised when the bounded native mapper rejects an invocation."""


def build_native_rec2020_interior(*, root: Path, output_dir: Path) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative=SOURCE,
        header_relative=HEADER,
        basename="nf_rec2020_oklab_interior_f32_v1",
    )


def load_native_rec2020_interior(dll_path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(Path(dll_path).resolve()))
    function = library.nf_rec2020_oklab_interior_f32_v1
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    function.restype = ctypes.c_int
    return library


def apply_native_rec2020_interior(
    library: ctypes.CDLL,
    pixels: np.ndarray,
    *,
    softness: float = 1.0 / 64.0,
    margin: float = 2.0 / 65535.0,
    iterations: int = 24,
    thread_count: int = 8,
    output: np.ndarray | None = None,
    chroma_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.float32
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or not pixels.flags.c_contiguous
    ):
        raise TypeError("pixels must be C-contiguous HxWx3 float32")
    if output is None:
        output = np.empty_like(pixels)
    if chroma_scale is None:
        chroma_scale = np.empty(pixels.shape[:2], dtype=np.float32)
    if (
        output.dtype != np.float32
        or output.shape != pixels.shape
        or not output.flags.c_contiguous
        or chroma_scale.dtype != np.float32
        or chroma_scale.shape != pixels.shape[:2]
        or not chroma_scale.flags.c_contiguous
    ):
        raise TypeError("native output buffers have invalid shape, dtype, or layout")
    status = library.nf_rec2020_oklab_interior_f32_v1(
        pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        chroma_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pixels.shape[0] * pixels.shape[1],
        float(softness),
        float(margin),
        int(iterations),
        int(thread_count),
    )
    if status != 0:
        raise NativeRec2020InteriorError(
            f"native Rec.2020 mapper rejected status {status}"
        )
    return output, chroma_scale


__all__ = [
    "NativeRec2020InteriorError",
    "apply_native_rec2020_interior",
    "build_native_rec2020_interior",
    "load_native_rec2020_interior",
]
