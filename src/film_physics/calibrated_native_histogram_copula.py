"""Source-observable finite-sample calibration for the P4HM native transport."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np

from src.film_physics.native_histogram_copula import apply_native_histogram_copula


def _spearman_from_gaussian(correlation: np.ndarray) -> np.ndarray:
    result = (6.0 / math.pi) * np.arcsin(correlation / 2.0)
    np.fill_diagonal(result, 1.0)
    return result


def _gaussian_from_spearman(correlation: np.ndarray) -> np.ndarray:
    result = 2.0 * np.sin(math.pi * correlation / 6.0)
    np.fill_diagonal(result, 1.0)
    return result


def apply_source_observable_calibrated_copula(
    library: ctypes.CDLL,
    fields: np.ndarray,
    *,
    profile_correlation: np.ndarray,
    rank_bins: int,
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Iteratively correct finite-sample rank correlation using source facts only."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    base = np.asarray(profile_correlation, dtype=np.float64)
    if base.shape != (3, 3) or not np.all(np.isfinite(base)):
        raise ValueError("invalid profile correlation")
    desired_uniform = _spearman_from_gaussian(base.copy())
    desired_gaussian = _gaussian_from_spearman(desired_uniform.copy())
    latent = base.copy()
    errors: list[float] = []
    diagnostics_rows: list[dict[str, Any]] = []
    output: np.ndarray | None = None
    for iteration in range(iterations):
        output, diagnostics = apply_native_histogram_copula(
            library,
            fields,
            target_correlation=latent,
            rank_bins=rank_bins,
        )
        observed = np.asarray(
            diagnostics["output_uniform_correlation"], dtype=np.float64
        )
        errors.append(float(np.max(np.abs(observed - desired_uniform))))
        diagnostics_rows.append(diagnostics)
        if iteration + 1 == iterations:
            break
        observed_gaussian = _gaussian_from_spearman(observed.copy())
        correction = desired_gaussian - observed_gaussian
        np.fill_diagonal(correction, 0.0)
        latent = latent + correction
        latent = 0.5 * (latent + latent.T)
        np.fill_diagonal(latent, 1.0)
        if float(np.min(np.linalg.eigvalsh(latent))) <= 0.0:
            raise RuntimeError("calibrated latent correlation is not positive definite")
    if output is None:
        raise AssertionError("calibrated native transport produced no output")
    return output, {
        "native_call_count": iterations,
        "iteration_maximum_correlation_errors": errors,
        "desired_uniform_correlation": desired_uniform.tolist(),
        "final_latent_correlation": latent.tolist(),
        "minimum_final_latent_eigenvalue": float(np.min(np.linalg.eigvalsh(latent))),
        "maximum_absolute_latent_correction": float(np.max(np.abs(latent - base))),
        "final_native_diagnostics": diagnostics_rows[-1],
    }


__all__ = ["apply_source_observable_calibrated_copula"]
