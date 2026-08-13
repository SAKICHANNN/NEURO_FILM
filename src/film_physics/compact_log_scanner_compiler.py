"""Typed compact log-response scanner compiler.

This is a deterministic execution primitive for a frozen synthetic compiler.
It does not identify or calibrate a physical scanner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompactLogScannerCompiler:
    compiler_id: str
    matrix_density_to_log10_rgb: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    bias_log10_rgb: tuple[float, float, float]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix_density_to_log10_rgb, dtype=np.float64)
        bias = np.asarray(self.bias_log10_rgb, dtype=np.float64)
        if not self.compiler_id or matrix.shape != (3, 3) or bias.shape != (3,):
            raise ValueError("invalid compact scanner compiler")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
            raise ValueError("compact scanner compiler must be finite")
        if np.any(matrix > 0.0):
            raise ValueError("density response must be nonincreasing")


def apply_compact_log_scanner(
    dye_density_yellow_magenta_cyan: np.ndarray,
    compiler: CompactLogScannerCompiler,
) -> np.ndarray:
    """Apply the frozen compiler in explicit float32 arithmetic."""

    density = np.asarray(dye_density_yellow_magenta_cyan)
    if density.dtype != np.float32:
        raise TypeError("compact scanner input must be float32")
    if density.ndim < 1 or density.shape[-1] != 3:
        raise ValueError("compact scanner input requires a final CMY dimension")
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("compact scanner density must be finite and nonnegative")
    matrix = np.asarray(compiler.matrix_density_to_log10_rgb, dtype=np.float32)
    bias = np.asarray(compiler.bias_log10_rgb, dtype=np.float32)
    log_response = density @ matrix + bias
    output = np.exp2(log_response * np.float32(np.log2(10.0))).astype(
        np.float32, copy=False
    )
    if not np.all(np.isfinite(output)) or np.any(output <= 0.0):
        raise RuntimeError("compact scanner output left the positive finite domain")
    return np.ascontiguousarray(output)


def interpret_negative_scan_relative(
    scanner_raw_rgb: np.ndarray,
    *,
    clear_scan_rgb: np.ndarray,
    maximum_density_scan_rgb: np.ndarray,
) -> np.ndarray:
    """Invert a scanner-RAW negative between two explicitly bound endpoints.

    Clear film maps to display white and the frozen maximum-density endpoint
    maps to display black.  No clipping is performed; values outside the
    endpoint interval are rejected.
    """

    values = np.asarray(scanner_raw_rgb)
    clear = np.asarray(clear_scan_rgb)
    maximum = np.asarray(maximum_density_scan_rgb)
    if values.dtype != np.float32 or clear.dtype != np.float32 or maximum.dtype != np.float32:
        raise TypeError("negative scan interpretation requires float32")
    if values.ndim < 1 or values.shape[-1] != 3 or clear.shape != (3,) or maximum.shape != (3,):
        raise ValueError("negative scan interpretation shape mismatch")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(clear)) or not np.all(np.isfinite(maximum)):
        raise ValueError("negative scan interpretation requires finite values")
    span = clear - maximum
    if np.any(span <= 0.0):
        raise ValueError("negative scan endpoints must have positive span")
    epsilon = np.float32(16.0 * np.finfo(np.float32).eps)
    if np.any(values > clear + epsilon) or np.any(values < maximum - epsilon):
        raise ValueError("negative scan value is outside bound endpoints")
    output = (values - maximum) / span
    output = np.ascontiguousarray(output, dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output < -epsilon) or np.any(output > 1.0 + epsilon):
        raise RuntimeError("negative scan interpretation left relative display domain")
    return output


def invert_compact_log_scanner(
    scanner_raw_rgb: np.ndarray,
    compiler: CompactLogScannerCompiler,
) -> np.ndarray:
    """Recover CMY dye amounts from positive scanner RGB without clipping."""

    values = np.asarray(scanner_raw_rgb)
    if values.dtype != np.float32:
        raise TypeError("compact scanner inverse input must be float32")
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("compact scanner inverse requires a final RGB dimension")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("compact scanner inverse requires positive finite values")
    matrix = np.asarray(compiler.matrix_density_to_log10_rgb, dtype=np.float64)
    bias = np.asarray(compiler.bias_log10_rgb, dtype=np.float64)
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1000.0:
        raise ValueError("compact scanner compiler is not safely invertible")
    log_response = np.log10(values.astype(np.float64)) - bias
    density = np.linalg.solve(matrix.T, log_response.T).T
    output = np.ascontiguousarray(density, dtype=np.float32)
    tolerance = np.float32(4e-6)
    if not np.all(np.isfinite(output)) or np.any(output < -tolerance):
        raise RuntimeError("compact scanner inverse left nonnegative density domain")
    return output


__all__ = [
    "CompactLogScannerCompiler",
    "apply_compact_log_scanner",
    "interpret_negative_scan_relative",
    "invert_compact_log_scanner",
]
