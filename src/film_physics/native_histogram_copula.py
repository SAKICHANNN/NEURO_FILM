"""ctypes binding for the P4HM native histogram-copula transport."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np


class NativeHistogramCopulaDiagnosticsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("abi_version", ctypes.c_uint32),
        ("rank_bins", ctypes.c_uint32),
        ("sample_count", ctypes.c_size_t),
        ("workspace_bytes", ctypes.c_size_t),
        ("input_normal_correlation", ctypes.c_double * 9),
        ("output_uniform_correlation", ctypes.c_double * 9),
        ("output_uniform_mean", ctypes.c_double * 3),
    ]


def load_native_histogram_copula_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_histogram_copula_f32_abi_version_v1.argtypes = []
    library.nf_histogram_copula_f32_abi_version_v1.restype = ctypes.c_uint32
    library.nf_histogram_copula_f32_workspace_bytes_v1.argtypes = [
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.nf_histogram_copula_f32_workspace_bytes_v1.restype = ctypes.c_int
    library.nf_histogram_copula_f32_apply_v1.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(NativeHistogramCopulaDiagnosticsV1),
    ]
    library.nf_histogram_copula_f32_apply_v1.restype = ctypes.c_int
    if library.nf_histogram_copula_f32_abi_version_v1() != 1:
        raise RuntimeError("native histogram copula ABI drift")
    return library


def apply_native_histogram_copula(
    library: ctypes.CDLL,
    fields: np.ndarray,
    *,
    target_correlation: np.ndarray,
    rank_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.ascontiguousarray(fields, dtype=np.float32)
    target = np.ascontiguousarray(target_correlation, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or target.shape != (3, 3):
        raise ValueError("invalid native histogram copula request")
    workspace_bytes = ctypes.c_size_t()
    status = library.nf_histogram_copula_f32_workspace_bytes_v1(
        values.shape[0], rank_bins, ctypes.byref(workspace_bytes)
    )
    if status != 0:
        raise RuntimeError(f"native histogram copula workspace failed: {status}")
    workspace = ctypes.create_string_buffer(workspace_bytes.value)
    output = np.empty_like(values)
    diagnostics = NativeHistogramCopulaDiagnosticsV1()
    diagnostics.struct_size = ctypes.sizeof(diagnostics)
    diagnostics.abi_version = 1
    status = library.nf_histogram_copula_f32_apply_v1(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.shape[0],
        target.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rank_bins,
        workspace,
        workspace_bytes.value,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.size,
        ctypes.byref(diagnostics),
    )
    if status != 0:
        raise RuntimeError(f"native histogram copula apply failed: {status}")
    return output, {
        "rank_bins": int(diagnostics.rank_bins),
        "sample_count": int(diagnostics.sample_count),
        "workspace_bytes": int(diagnostics.workspace_bytes),
        "input_normal_correlation": np.ctypeslib.as_array(
            diagnostics.input_normal_correlation
        ).reshape(3, 3).tolist(),
        "output_uniform_correlation": np.ctypeslib.as_array(
            diagnostics.output_uniform_correlation
        ).reshape(3, 3).tolist(),
        "output_uniform_mean": list(diagnostics.output_uniform_mean),
    }


__all__ = [
    "NativeHistogramCopulaDiagnosticsV1",
    "apply_native_histogram_copula",
    "load_native_histogram_copula_library",
]
