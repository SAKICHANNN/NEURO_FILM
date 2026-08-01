"""Canonical float32 execution for explicit density-to-print interpretation."""

from __future__ import annotations

import numpy as np

from src.roll2film.sensitometry_print import DensityToPrintInterpretation


def _raw_float32(
    operator: DensityToPrintInterpretation,
    density: np.ndarray,
) -> np.ndarray:
    absorption = density @ np.asarray(
        operator.dye_absorption_matrix, dtype=np.float32
    ).T
    transmission = np.power(np.float32(10.0), -absorption)
    exposure = transmission @ np.asarray(operator.print_matrix, dtype=np.float32).T
    log_exposure = np.log2(exposure + np.float32(operator.exposure_floor))
    midpoint = np.asarray(operator.paper_midpoints, dtype=np.float32)
    slope = np.asarray(operator.paper_slopes, dtype=np.float32)
    maximum = np.asarray(operator.paper_maximum_densities, dtype=np.float32)
    sigmoid = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-slope * (log_exposure - midpoint))
    )
    paper_density = maximum * sigmoid
    return np.power(np.float32(10.0), -paper_density)


def apply_density_to_print_float32(
    operator: DensityToPrintInterpretation,
    layer_density: np.ndarray,
) -> np.ndarray:
    """Apply all explicit print stages in canonical float32 without clipping."""
    if not isinstance(operator, DensityToPrintInterpretation):
        raise TypeError("operator must be DensityToPrintInterpretation")
    density = np.asarray(layer_density, dtype=np.float32)
    black = np.asarray(operator.black_reference_density, dtype=np.float32)
    white = np.asarray(operator.white_reference_density, dtype=np.float32)
    if (
        density.ndim < 2
        or density.shape[-1] != 3
        or not np.all(np.isfinite(density))
        or np.any(density < black)
        or np.any(density > white)
    ):
        raise ValueError("layer density falls outside declared float32 references")
    endpoints = _raw_float32(operator, np.stack((black, white), axis=0))
    output = (_raw_float32(operator, density) - endpoints[0]) / (
        endpoints[1] - endpoints[0]
    )
    if (
        not np.all(np.isfinite(output))
        or np.any(output < np.float32(-1e-6))
        or np.any(output > np.float32(1.0 + 1e-6))
    ):
        raise RuntimeError("float32 print interpretation escaped normalized endpoints")
    return np.asarray(output, dtype=np.float32)


__all__ = ["apply_density_to_print_float32"]
