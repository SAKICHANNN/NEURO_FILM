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


__all__ = ["CompactLogScannerCompiler", "apply_compact_log_scanner"]
