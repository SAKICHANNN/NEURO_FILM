"""Continuous NPS of a Gaussian-perturbed lattice with a Gaussian mark."""

from __future__ import annotations

import math

import numpy as np


def perturbed_lattice_gaussian_mark_nps(
    frequency_lines_per_mm: np.ndarray | float,
    particle_sigma_millimetres: float,
    jitter_sigma_millimetres: float,
) -> np.ndarray:
    """Return the positive continuous diffuse spectrum, excluding Bragg atoms.

    For independent isotropic Gaussian displacements, the continuous part of
    the perturbed-lattice structure factor is ``1 - |phi_p(k)|^2``.  A
    Gaussian developed-particle optical footprint supplies the mark power.
    Overall intensity and mark amplitude are intentionally left to the
    evaluator's one positive per-state scale.
    """

    frequency = np.asarray(frequency_lines_per_mm, dtype=np.float64)
    particle_sigma = float(particle_sigma_millimetres)
    jitter_sigma = float(jitter_sigma_millimetres)
    if (
        not np.all(np.isfinite(frequency))
        or np.any(frequency < 0.0)
        or not math.isfinite(particle_sigma)
        or particle_sigma <= 0.0
        or not math.isfinite(jitter_sigma)
        or jitter_sigma <= 0.0
    ):
        raise ValueError("invalid perturbed-lattice NPS arguments")
    angular_squared = np.square(2.0 * math.pi * frequency)
    mark_power = np.exp(-angular_squared * particle_sigma * particle_sigma)
    diffuse_structure = -np.expm1(
        -angular_squared * jitter_sigma * jitter_sigma
    )
    spectrum = mark_power * diffuse_structure
    if (
        not np.all(np.isfinite(spectrum))
        or np.any(spectrum < 0.0)
        or np.any((frequency > 0.0) & (spectrum <= 0.0))
    ):
        raise RuntimeError("perturbed-lattice diffuse NPS escaped its domain")
    return spectrum


__all__ = ["perturbed_lattice_gaussian_mark_nps"]
