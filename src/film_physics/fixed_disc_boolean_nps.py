"""Clean-room fixed-disc Boolean-model covariance and radial spectrum.

The equations are independently implemented from the stochastic-geometry
definition used by Newson et al. (IPOL 2017).  No implementation code from
the GPL reference package is used here.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import j0


def fixed_disc_overlap_area(
    distance_millimetres: np.ndarray | float,
    radius_millimetres: float,
) -> np.ndarray:
    """Return the intersection area of two equal discs at each separation."""

    distance = np.asarray(distance_millimetres, dtype=np.float64)
    radius = float(radius_millimetres)
    if (
        not math.isfinite(radius)
        or radius <= 0.0
        or not np.all(np.isfinite(distance))
        or np.any(distance < 0.0)
    ):
        raise ValueError("disc radius and distances must be finite and positive")
    ratio = np.clip(distance / (2.0 * radius), 0.0, 1.0)
    radicand = np.maximum(0.0, 4.0 * radius * radius - np.square(distance))
    overlap = (
        2.0 * radius * radius * np.arccos(ratio)
        - 0.5 * distance * np.sqrt(radicand)
    )
    return np.where(distance <= 2.0 * radius, overlap, 0.0)


def fixed_disc_boolean_covariance(
    distance_millimetres: np.ndarray | float,
    radius_millimetres: float,
    coverage_probability: float,
) -> np.ndarray:
    """Return covariance of the covered-set indicator for Poisson discs."""

    coverage = float(coverage_probability)
    radius = float(radius_millimetres)
    if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
        raise ValueError("coverage probability must lie strictly inside (0, 1)")
    overlap = fixed_disc_overlap_area(distance_millimetres, radius)
    germ_intensity = -math.log1p(-coverage) / (math.pi * radius * radius)
    return (1.0 - coverage) ** 2 * np.expm1(germ_intensity * overlap)


def fixed_disc_boolean_radial_nps(
    frequency_lines_per_mm: np.ndarray | float,
    radius_millimetres: float,
    coverage_probability: float,
    *,
    quadrature_samples: int = 4097,
) -> np.ndarray:
    """Hankel-transform the compact Boolean covariance to a radial 2-D NPS."""

    frequency = np.asarray(frequency_lines_per_mm, dtype=np.float64)
    if (
        not np.all(np.isfinite(frequency))
        or np.any(frequency < 0.0)
        or quadrature_samples < 257
        or quadrature_samples % 2 != 1
    ):
        raise ValueError("invalid radial NPS frequency or quadrature")
    radius = float(radius_millimetres)
    distance = np.linspace(0.0, 2.0 * radius, quadrature_samples, dtype=np.float64)
    step = float(distance[1] - distance[0])
    weights = np.ones(quadrature_samples, dtype=np.float64)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    weights *= step / 3.0
    covariance = fixed_disc_boolean_covariance(
        distance, radius, coverage_probability
    )
    kernel = j0(
        2.0
        * math.pi
        * frequency.reshape(-1, 1)
        * distance.reshape(1, -1)
    )
    spectrum = 2.0 * math.pi * (kernel @ (covariance * distance * weights))
    spectrum = spectrum.reshape(frequency.shape)
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum <= 0.0):
        raise RuntimeError("fixed-disc Boolean NPS is not finite and positive")
    return spectrum


__all__ = [
    "fixed_disc_boolean_covariance",
    "fixed_disc_boolean_radial_nps",
    "fixed_disc_overlap_area",
]
