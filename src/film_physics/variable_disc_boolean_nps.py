"""Clean-room bounded-lognormal Poisson-disc Boolean radial spectra."""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import j0, ndtri


def bounded_lognormal_radius_quadrature(
    median_radius_millimetres: float,
    log_radius_sigma: float,
    *,
    cdf_interval: tuple[float, float] = (1e-5, 0.99999),
    nodes: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized deterministic quadrature for a truncated lognormal."""

    median = float(median_radius_millimetres)
    sigma = float(log_radius_sigma)
    lower, upper = (float(value) for value in cdf_interval)
    if (
        not math.isfinite(median)
        or median <= 0.0
        or not math.isfinite(sigma)
        or sigma <= 0.0
        or not 0.0 < lower < upper < 1.0
        or nodes < 8
    ):
        raise ValueError("invalid bounded-lognormal radius quadrature")
    abscissa, base_weights = leggauss(nodes)
    probability = lower + (abscissa + 1.0) * 0.5 * (upper - lower)
    radii = median * np.exp(sigma * ndtri(probability))
    weights = base_weights * 0.5
    weights /= np.sum(weights)
    if (
        not np.all(np.isfinite(radii))
        or not np.all(radii > 0.0)
        or not np.all(weights > 0.0)
    ):
        raise RuntimeError("bounded-lognormal quadrature escaped its domain")
    return radii, weights


def variable_disc_boolean_covariance(
    distance_millimetres: np.ndarray | float,
    median_radius_millimetres: float,
    log_radius_sigma: float,
    coverage_probability: float,
    *,
    cdf_interval: tuple[float, float] = (1e-5, 0.99999),
    radius_nodes: int = 32,
) -> np.ndarray:
    """Return Boolean covered-indicator covariance averaged over disc radii."""

    distance = np.asarray(distance_millimetres, dtype=np.float64)
    coverage = float(coverage_probability)
    if (
        not np.all(np.isfinite(distance))
        or np.any(distance < 0.0)
        or not math.isfinite(coverage)
        or not 0.0 < coverage < 1.0
    ):
        raise ValueError("invalid variable-disc covariance arguments")
    radii, weights = bounded_lognormal_radius_quadrature(
        median_radius_millimetres,
        log_radius_sigma,
        cdf_interval=cdf_interval,
        nodes=radius_nodes,
    )
    d = distance.reshape(-1, 1)
    r = radii.reshape(1, -1)
    ratio = np.clip(d / (2.0 * r), 0.0, 1.0)
    overlap = 2.0 * np.square(r) * np.arccos(ratio)
    overlap -= 0.5 * d * np.sqrt(np.maximum(0.0, 4.0 * np.square(r) - np.square(d)))
    overlap = np.where(d <= 2.0 * r, overlap, 0.0)
    mean_overlap = overlap @ weights
    mean_area = math.pi * float(np.sum(weights * np.square(radii)))
    germ_intensity = -math.log1p(-coverage) / mean_area
    covariance = (1.0 - coverage) ** 2 * np.expm1(germ_intensity * mean_overlap)
    return covariance.reshape(distance.shape)


def variable_disc_boolean_radial_nps(
    frequency_lines_per_mm: np.ndarray | float,
    median_radius_millimetres: float,
    log_radius_sigma: float,
    coverage_probability: float,
    *,
    cdf_interval: tuple[float, float] = (1e-5, 0.99999),
    radius_nodes: int = 32,
    radial_samples: int = 4097,
) -> np.ndarray:
    """Hankel-transform a bounded-lognormal Boolean covariance."""

    frequency = np.asarray(frequency_lines_per_mm, dtype=np.float64)
    if (
        not np.all(np.isfinite(frequency))
        or np.any(frequency < 0.0)
        or radial_samples < 257
        or radial_samples % 2 != 1
    ):
        raise ValueError("invalid variable-disc radial NPS arguments")
    radii, _ = bounded_lognormal_radius_quadrature(
        median_radius_millimetres,
        log_radius_sigma,
        cdf_interval=cdf_interval,
        nodes=radius_nodes,
    )
    distance = np.linspace(0.0, 2.0 * float(np.max(radii)), radial_samples)
    step = float(distance[1] - distance[0])
    weights = np.ones(radial_samples, dtype=np.float64)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    weights *= step / 3.0
    covariance = variable_disc_boolean_covariance(
        distance,
        median_radius_millimetres,
        log_radius_sigma,
        coverage_probability,
        cdf_interval=cdf_interval,
        radius_nodes=radius_nodes,
    )
    kernel = j0(2.0 * math.pi * frequency.reshape(-1, 1) * distance.reshape(1, -1))
    spectrum = 2.0 * math.pi * (kernel @ (covariance * distance * weights))
    spectrum = spectrum.reshape(frequency.shape)
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum <= 0.0):
        raise RuntimeError("variable-disc Boolean NPS is not finite and positive")
    return spectrum


__all__ = [
    "bounded_lognormal_radius_quadrature",
    "variable_disc_boolean_covariance",
    "variable_disc_boolean_radial_nps",
]
