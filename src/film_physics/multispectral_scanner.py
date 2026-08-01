"""Explicit ideal-band multispectral scanner compiler primitives.

The primitives intentionally model ideal centre samples rather than real LED
spectral power distributions.  They preserve a bounded, inspectable path from
spectral transmittance samples to relative D65 XYZ and never imply a measured
scanner profile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


@dataclass(frozen=True)
class IdealBandProfile:
    profile_id: str
    center_nm: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("ideal-band profile_id must be non-empty")
        centers = np.asarray(self.center_nm, dtype=np.float64)
        if (
            centers.ndim != 1
            or centers.size < 3
            or not np.all(np.isfinite(centers))
            or np.any(np.diff(centers) <= 0.0)
        ):
            raise ValueError(
                "ideal-band centres must contain at least three finite, "
                "strictly increasing wavelengths"
            )

    @property
    def profile_sha256(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "center_nm": list(self.center_nm),
            "observation": "piecewise-linear-ideal-delta-centre-v1",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest()


def sample_ideal_bands(
    spectra: np.ndarray,
    wavelength_nm: np.ndarray,
    profile: IdealBandProfile,
) -> np.ndarray:
    """Sample spectra at fixed ideal centres using explicit linear weights."""

    values = np.asarray(spectra)
    wavelength = np.asarray(wavelength_nm)
    if values.dtype != np.float64 or wavelength.dtype != np.float64:
        raise TypeError("ideal-band sampling requires float64 inputs")
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError("wavelength grid must be one-dimensional")
    if (
        not np.all(np.isfinite(wavelength))
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength grid must be finite and increasing")
    if values.ndim < 2 or values.shape[-1] != wavelength.size:
        raise ValueError("spectral wavelength dimension mismatch")
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("spectra must be finite in [0, 1]")
    centers = np.asarray(profile.center_nm, dtype=np.float64)
    if centers[0] < wavelength[0] or centers[-1] > wavelength[-1]:
        raise ValueError("ideal-band centres must lie inside the wavelength grid")

    upper = np.searchsorted(wavelength, centers, side="right")
    upper = np.minimum(upper, wavelength.size - 1)
    lower = np.maximum(upper - 1, 0)
    exact = wavelength[lower] == centers
    upper[exact] = lower[exact]
    span = wavelength[upper] - wavelength[lower]
    alpha = np.divide(
        centers - wavelength[lower],
        span,
        out=np.zeros_like(centers),
        where=span > 0.0,
    )
    return (
        values[..., lower] * (1.0 - alpha)
        + values[..., upper] * alpha
    )


def fit_nonnegative_xyz_compiler(
    band_values: np.ndarray,
    target_xyz: np.ndarray,
    clear_xyz: np.ndarray,
) -> np.ndarray:
    """Fit a shared nonnegative band-to-XYZ matrix with exact clear response."""

    bands = np.asarray(band_values)
    target = np.asarray(target_xyz)
    clear = np.asarray(clear_xyz)
    if bands.dtype != np.float64 or target.dtype != np.float64:
        raise TypeError("multispectral compiler fitting requires float64 arrays")
    if (
        bands.ndim != 2
        or bands.shape[0] < bands.shape[1]
        or target.shape != (bands.shape[0], 3)
    ):
        raise ValueError("compiler fit arrays have invalid shapes")
    if clear.shape != (3,):
        raise ValueError("clear XYZ must be a three-vector")
    if (
        not np.all(np.isfinite(bands))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(clear))
        or np.any(bands < 0.0)
        or np.any(bands > 1.0)
        or np.any(target < 0.0)
        or np.any(clear <= 0.0)
    ):
        raise ValueError("compiler fit inputs must be finite and physical")
    columns = []
    for channel in range(3):
        weights, _ = nnls(bands, target[:, channel])
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("nonnegative compiler fit produced an empty column")
        columns.append(weights * (float(clear[channel]) / total))
    matrix = np.stack(columns, axis=1)
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise AssertionError("compiler matrix left the nonnegative domain")
    return matrix


def apply_xyz_compiler(
    band_values: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Apply a fixed nonnegative multispectral compiler without clipping."""

    bands = np.asarray(band_values)
    transform = np.asarray(matrix)
    if bands.dtype != np.float64 or transform.dtype != np.float64:
        raise TypeError("multispectral compiler application requires float64")
    if bands.ndim != 2 or transform.shape != (bands.shape[1], 3):
        raise ValueError("compiler application arrays have invalid shapes")
    if (
        not np.all(np.isfinite(bands))
        or not np.all(np.isfinite(transform))
        or np.any(bands < 0.0)
        or np.any(bands > 1.0)
        or np.any(transform < 0.0)
    ):
        raise ValueError("compiler application inputs must be finite and bounded")
    output = bands @ transform
    if not np.all(np.isfinite(output)) or np.any(output < 0.0):
        raise AssertionError("compiler output left the nonnegative domain")
    return output


__all__ = [
    "IdealBandProfile",
    "apply_xyz_compiler",
    "fit_nonnegative_xyz_compiler",
    "sample_ideal_bands",
]
