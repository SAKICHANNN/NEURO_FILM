"""ctypes binding for the P4HS fast inverse-Gamma primitive."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np


class NativeFastGammaDensityDiagnosticsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("abi_version", ctypes.c_uint32),
        ("direct_iterations", ctypes.c_uint32),
        ("newton_iterations", ctypes.c_uint32),
        ("sample_count", ctypes.c_size_t),
        ("direct_branch_count", ctypes.c_size_t),
        ("newton_branch_count", ctypes.c_size_t),
        ("asymptotic_branch_count", ctypes.c_size_t),
        ("minimum_output", ctypes.c_double),
        ("maximum_output", ctypes.c_double),
    ]


def load_native_fast_gamma_density_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_gamma_density_fast_f64_abi_version_v1.restype = ctypes.c_uint32
    library.nf_gamma_density_fast_f64_apply_v1.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(NativeFastGammaDensityDiagnosticsV1),
    ]
    library.nf_gamma_density_fast_f64_apply_v1.restype = ctypes.c_int
    if library.nf_gamma_density_fast_f64_abi_version_v1() != 1:
        raise RuntimeError("native fast gamma ABI drift")
    return library


def apply_native_fast_gamma_density(
    library: ctypes.CDLL,
    uniforms: np.ndarray,
    shapes: np.ndarray,
    scales: np.ndarray,
    *,
    direct_iterations: int,
    newton_iterations: int,
    direct_shape_upper: float,
    newton_shape_upper: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    probability = np.ascontiguousarray(uniforms, dtype=np.float64)
    shape = np.ascontiguousarray(shapes, dtype=np.float64)
    scale = np.ascontiguousarray(scales, dtype=np.float64)
    if (
        probability.ndim != 1
        or shape.shape != probability.shape
        or scale.shape != probability.shape
    ):
        raise ValueError("invalid native fast gamma request")
    output = np.empty_like(probability)
    diagnostics = NativeFastGammaDensityDiagnosticsV1()
    diagnostics.struct_size = ctypes.sizeof(diagnostics)
    diagnostics.abi_version = 1
    status = library.nf_gamma_density_fast_f64_apply_v1(
        probability.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        shape.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        scale.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        probability.size,
        direct_iterations,
        newton_iterations,
        direct_shape_upper,
        newton_shape_upper,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        output.size,
        ctypes.byref(diagnostics),
    )
    if status != 0:
        raise RuntimeError(f"native fast gamma apply failed: {status}")
    return output, {
        "direct_iterations": int(diagnostics.direct_iterations),
        "newton_iterations": int(diagnostics.newton_iterations),
        "sample_count": int(diagnostics.sample_count),
        "direct_branch_count": int(diagnostics.direct_branch_count),
        "newton_branch_count": int(diagnostics.newton_branch_count),
        "asymptotic_branch_count": int(diagnostics.asymptotic_branch_count),
        "minimum_output": float(diagnostics.minimum_output),
        "maximum_output": float(diagnostics.maximum_output),
    }


__all__ = [
    "NativeFastGammaDensityDiagnosticsV1",
    "apply_native_fast_gamma_density",
    "load_native_fast_gamma_density_library",
]
