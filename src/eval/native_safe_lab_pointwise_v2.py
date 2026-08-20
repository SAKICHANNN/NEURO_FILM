"""Host-native v2 exact-semantics boundary for pointwise safe-Lab stages."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.safe_lab import SafeLabSourceContext
from src.eval.native_msvc import build_msvc_c11_dll

SOURCE = "native/color_engine/nf_safe_lab_pointwise_f32_v2.c"
HEADER = "native/color_engine/nf_safe_lab_pointwise_f32_v2.h"


class NativeSafeLabPointwiseV2Error(RuntimeError):
    """Raised when the native pointwise safe-Lab kernel rejects an invocation."""


def build_native_safe_lab_pointwise_v2(*, root: Path, output_dir: Path) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative=SOURCE,
        header_relative=HEADER,
        basename="nf_safe_lab_pointwise_f32_v2",
    )


def load_native_safe_lab_pointwise_v2(dll_path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(Path(dll_path).resolve()))
    function = library.nf_safe_lab_pointwise_f32_v2
    float_pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [
        float_pointer,
        float_pointer,
        ctypes.c_uint64,
        float_pointer,
        float_pointer,
        float_pointer,
        float_pointer,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_double,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint32,
    ]
    function.restype = ctypes.c_int
    return library


def apply_native_safe_lab_pointwise_v2(
    library: ctypes.CDLL,
    source_lab: np.ndarray,
    *,
    source_context: SafeLabSourceContext,
    destination_mean: np.ndarray,
    destination_std: np.ndarray,
    strength: float,
    luma_strength: float,
    chroma_curve_strength: float,
    neutral_protect: float,
    skin_protect: float,
    max_chroma_gain: float,
    max_chroma_boost: float,
    max_chroma_absolute: float | None,
    thread_count: int = 8,
    output: np.ndarray | None = None,
) -> np.ndarray:
    if (
        not isinstance(source_lab, np.ndarray)
        or source_lab.dtype != np.float32
        or source_lab.ndim != 3
        or source_lab.shape[2] != 3
        or not source_lab.flags.c_contiguous
    ):
        raise TypeError("source_lab must be C-contiguous HxWx3 float32")
    if output is None:
        output = np.empty_like(source_lab)
    if (
        output.dtype != np.float32
        or output.shape != source_lab.shape
        or not output.flags.c_contiguous
    ):
        raise TypeError("native output buffer has invalid shape, dtype, or layout")
    source_mean = np.ascontiguousarray(source_context.lab_mean, dtype=np.float32)
    source_std = np.ascontiguousarray(source_context.lab_std, dtype=np.float32)
    destination_mean = np.ascontiguousarray(destination_mean, dtype=np.float32)
    destination_std = np.ascontiguousarray(destination_std, dtype=np.float32)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = library.nf_safe_lab_pointwise_f32_v2(
        source_lab.ctypes.data_as(pointer),
        output.ctypes.data_as(pointer),
        source_lab.shape[0] * source_lab.shape[1],
        source_mean.ctypes.data_as(pointer),
        source_std.ctypes.data_as(pointer),
        destination_mean.ctypes.data_as(pointer),
        destination_std.ctypes.data_as(pointer),
        float(strength),
        float(luma_strength) * float(strength),
        float(chroma_curve_strength),
        float(neutral_protect),
        ctypes.c_double(float(skin_protect)),
        float(max_chroma_gain),
        float(max_chroma_boost),
        float(max_chroma_absolute) if max_chroma_absolute is not None else -1.0,
        int(thread_count),
    )
    if status != 0:
        raise NativeSafeLabPointwiseV2Error(
            f"native pointwise safe-Lab rejected status {status}"
        )
    return output


__all__ = [
    "NativeSafeLabPointwiseV2Error",
    "apply_native_safe_lab_pointwise_v2",
    "build_native_safe_lab_pointwise_v2",
    "load_native_safe_lab_pointwise_v2",
]
