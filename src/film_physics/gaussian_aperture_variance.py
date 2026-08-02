"""Continuous circular-aperture variance for a Gaussian covariance field."""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss


def gaussian_covariance_disk_average_variance(
    aperture_diameter_micrometres: np.ndarray | float,
    correlation_scale_micrometres: np.ndarray | float,
    *,
    quadrature_order: int = 512,
) -> np.ndarray:
    """Return variance of a unit-variance Gaussian field averaged by a disk.

    The integral uses the exact overlap area of two equal circular apertures.
    It is a physical-domain variance operator, not a raster approximation.
    """

    diameters = np.asarray(aperture_diameter_micrometres, dtype=np.float64)
    scales = np.asarray(correlation_scale_micrometres, dtype=np.float64)
    if (
        not np.all(np.isfinite(diameters))
        or not np.all(np.isfinite(scales))
        or np.any(diameters <= 0.0)
        or np.any(scales <= 0.0)
        or isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, int)
        or quadrature_order < 64
        or quadrature_order > 4096
    ):
        raise ValueError("invalid Gaussian aperture-variance request")
    nodes, weights = leggauss(quadrature_order)
    x = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    overlap = np.arccos(x) - x * np.sqrt(np.maximum(1.0 - np.square(x), 0.0))
    radii = 0.5 * diameters
    ratio = np.expand_dims(radii / scales, axis=-1)
    integrand = (
        x
        * overlap
        * np.exp(-2.0 * np.square(ratio) * np.square(x))
    )
    variance = (16.0 / math.pi) * np.sum(integrand * weights, axis=-1)
    if (
        not np.all(np.isfinite(variance))
        or np.any(variance <= 0.0)
        or np.any(variance > 1.0 + 1e-12)
    ):
        raise RuntimeError("Gaussian aperture variance left its analytic domain")
    return np.minimum(variance, 1.0)


def gaussian_covariance_aperture_rms_ratio(
    aperture_diameter_micrometres: np.ndarray | float,
    *,
    reference_aperture_micrometres: float,
    correlation_scale_micrometres: np.ndarray | float,
    quadrature_order: int = 512,
) -> np.ndarray:
    """Return aperture RMS relative to one reference aperture."""

    if (
        not math.isfinite(reference_aperture_micrometres)
        or reference_aperture_micrometres <= 0.0
    ):
        raise ValueError("reference aperture must be finite and positive")
    numerator = gaussian_covariance_disk_average_variance(
        aperture_diameter_micrometres,
        correlation_scale_micrometres,
        quadrature_order=quadrature_order,
    )
    denominator = gaussian_covariance_disk_average_variance(
        reference_aperture_micrometres,
        correlation_scale_micrometres,
        quadrature_order=quadrature_order,
    )
    return np.sqrt(numerator / denominator)


__all__ = [
    "gaussian_covariance_aperture_rms_ratio",
    "gaussian_covariance_disk_average_variance",
]
