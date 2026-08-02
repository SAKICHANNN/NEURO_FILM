"""Explicit Gaussian-mark Thomas cluster-process radial NPS."""

from __future__ import annotations

import math

import numpy as np


def thomas_cluster_gaussian_mark_nps(
    frequency_lines_per_mm: np.ndarray | float,
    particle_sigma_millimetres: float,
    cluster_sigma_millimetres: float,
    mean_offspring_per_cluster: float,
) -> np.ndarray:
    """Return a unit-amplitude marked Thomas-process spectrum.

    The point-process structure factor follows Hawat et al. (2023), Equation
    8. A Gaussian developed-particle optical footprint supplies the explicit
    mark transfer function. Overall point intensity and mark amplitude are
    intentionally left to the evaluator's single positive scale.
    """

    frequency = np.asarray(frequency_lines_per_mm, dtype=np.float64)
    particle_sigma = float(particle_sigma_millimetres)
    cluster_sigma = float(cluster_sigma_millimetres)
    offspring = float(mean_offspring_per_cluster)
    if (
        not np.all(np.isfinite(frequency))
        or np.any(frequency < 0.0)
        or not math.isfinite(particle_sigma)
        or particle_sigma <= 0.0
        or not math.isfinite(cluster_sigma)
        or cluster_sigma <= 0.0
        or not math.isfinite(offspring)
        or offspring <= 0.0
    ):
        raise ValueError("invalid Thomas-cluster NPS arguments")
    angular_squared = np.square(2.0 * math.pi * frequency)
    mark_power = np.exp(-angular_squared * particle_sigma * particle_sigma)
    structure_factor = 1.0 + offspring * np.exp(
        -angular_squared * cluster_sigma * cluster_sigma
    )
    spectrum = mark_power * structure_factor
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum <= 0.0):
        raise RuntimeError("Thomas-cluster NPS escaped its positive domain")
    return spectrum


__all__ = ["thomas_cluster_gaussian_mark_nps"]
