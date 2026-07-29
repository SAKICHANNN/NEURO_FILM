"""Float64 spectral scanner reference primitives.

This module models only the acquisition integral

    scanner_c = integral(illuminant * film_transmittance * sensitivity_c)

normalized by the clear-film response.  It is an offline reference boundary,
not a measured scanner profile or a product colour transform.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


def _integration_weights(wavelength_nm: np.ndarray) -> np.ndarray:
    intervals = np.diff(wavelength_nm)
    weights = np.empty_like(wavelength_nm)
    weights[0] = intervals[0] * 0.5
    weights[-1] = intervals[-1] * 0.5
    weights[1:-1] = (intervals[:-1] + intervals[1:]) * 0.5
    return weights


@dataclass(frozen=True)
class SpectralScannerProfile:
    profile_id: str
    wavelength_nm: tuple[float, ...]
    illuminant_spd: tuple[float, ...]
    sensor_sensitivity_rgb: tuple[
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("spectral scanner profile_id must be non-empty")
        wavelength = np.asarray(self.wavelength_nm, dtype=np.float64)
        illuminant = np.asarray(self.illuminant_spd, dtype=np.float64)
        sensitivity = np.asarray(self.sensor_sensitivity_rgb, dtype=np.float64)
        if wavelength.ndim != 1 or wavelength.size < 7:
            raise ValueError("spectral scanner requires at least seven wavelengths")
        if (
            not np.all(np.isfinite(wavelength))
            or np.any(np.diff(wavelength) <= 0.0)
        ):
            raise ValueError("wavelengths must be finite and strictly increasing")
        if illuminant.shape != wavelength.shape or (
            not np.all(np.isfinite(illuminant))
            or np.any(illuminant < 0.0)
            or not np.any(illuminant > 0.0)
        ):
            raise ValueError("illuminant must be finite, nonnegative and nonzero")
        if sensitivity.shape != (3, wavelength.size) or (
            not np.all(np.isfinite(sensitivity))
            or np.any(sensitivity < 0.0)
            or np.any(np.sum(sensitivity, axis=1) <= 0.0)
        ):
            raise ValueError(
                "sensor sensitivity must be finite nonnegative RGB x wavelength"
            )
        if np.linalg.matrix_rank(self.response_weights) < 3:
            raise ValueError("spectral scanner RGB responses must be independent")

    @property
    def response_weights(self) -> np.ndarray:
        wavelength = np.asarray(self.wavelength_nm, dtype=np.float64)
        illuminant = np.asarray(self.illuminant_spd, dtype=np.float64)
        sensitivity = np.asarray(self.sensor_sensitivity_rgb, dtype=np.float64)
        raw = (
            sensitivity
            * illuminant[None, :]
            * _integration_weights(wavelength)[None, :]
        )
        return raw / np.sum(raw, axis=1, keepdims=True)

    @property
    def profile_sha256(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "wavelength_nm": list(self.wavelength_nm),
            "illuminant_spd": list(self.illuminant_spd),
            "sensor_sensitivity_rgb": [
                list(row) for row in self.sensor_sensitivity_rgb
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest()


def _gaussian(
    wavelength_nm: np.ndarray, center_nm: float, sigma_nm: float
) -> np.ndarray:
    if (
        not math.isfinite(center_nm)
        or not math.isfinite(sigma_nm)
        or sigma_nm <= 0.0
    ):
        raise ValueError("Gaussian center/sigma must be finite and sigma positive")
    return np.exp(-0.5 * ((wavelength_nm - center_nm) / sigma_nm) ** 2)


def synthetic_profile_from_contract(
    wavelength_nm: np.ndarray,
    row: dict[str, Any],
) -> SpectralScannerProfile:
    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    illuminant = np.zeros_like(wavelength)
    for component in row["illuminant_components"]:
        weight = float(component["weight"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("illuminant weights must be finite and positive")
        illuminant += weight * _gaussian(
            wavelength,
            float(component["center_nm"]),
            float(component["sigma_nm"]),
        )
    centers = tuple(float(value) for value in row["sensor_center_nm_rgb"])
    sigmas = tuple(float(value) for value in row["sensor_sigma_nm_rgb"])
    if len(centers) != 3 or len(sigmas) != 3:
        raise ValueError("sensor centers and sigmas must contain RGB triples")
    sensitivity = tuple(
        tuple(float(value) for value in _gaussian(wavelength, center, sigma))
        for center, sigma in zip(centers, sigmas, strict=True)
    )
    return SpectralScannerProfile(
        profile_id=str(row["profile_id"]),
        wavelength_nm=tuple(float(value) for value in wavelength),
        illuminant_spd=tuple(float(value) for value in illuminant),
        sensor_sensitivity_rgb=sensitivity,
    )


def apply_spectral_scanner(
    transmittance: np.ndarray,
    profile: SpectralScannerProfile,
) -> np.ndarray:
    values = np.asarray(transmittance)
    if values.dtype != np.float64:
        raise TypeError("spectral scanner reference input must be float64")
    if values.ndim < 1 or values.shape[-1] != len(profile.wavelength_nm):
        raise ValueError("spectral transmittance wavelength dimension mismatch")
    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("spectral transmittance must be finite in (0, 1]")
    return np.tensordot(values, profile.response_weights.T, axes=([-1], [0]))


def construct_primary_metamer_pair(
    primary: SpectralScannerProfile,
    observer: SpectralScannerProfile,
    *,
    neutral_transmittance: float,
    perturbation_max_abs: float,
) -> tuple[np.ndarray, np.ndarray]:
    if primary.wavelength_nm != observer.wavelength_nm:
        raise ValueError("metamer profiles must share a wavelength grid")
    if (
        not math.isfinite(neutral_transmittance)
        or not 0.0 < neutral_transmittance < 1.0
        or not math.isfinite(perturbation_max_abs)
        or perturbation_max_abs <= 0.0
        or perturbation_max_abs >= min(
            neutral_transmittance, 1.0 - neutral_transmittance
        )
    ):
        raise ValueError("metamer perturbation must stay strictly in transmittance")
    primary_weights = primary.response_weights
    observer_weights = observer.response_weights
    gram = primary_weights @ primary_weights.T
    projector = np.eye(primary_weights.shape[1], dtype=np.float64) - (
        primary_weights.T @ np.linalg.solve(gram, primary_weights)
    )
    candidates = [projector @ row for row in observer_weights]
    direction = max(
        candidates,
        key=lambda row: float(np.linalg.norm(observer_weights @ row)),
    )
    maximum = float(np.max(np.abs(direction)))
    if maximum <= np.finfo(np.float64).eps:
        raise ValueError("observer supplies no primary-metamer separation")
    direction = direction * (perturbation_max_abs / maximum)
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    neutral = np.full(direction.shape, neutral_transmittance, dtype=np.float64)
    first = neutral + direction
    second = neutral - direction
    if np.any(first <= 0.0) or np.any(first > 1.0):
        raise AssertionError("constructed first metamer left transmittance domain")
    if np.any(second <= 0.0) or np.any(second > 1.0):
        raise AssertionError("constructed second metamer left transmittance domain")
    return first, second


__all__ = [
    "SpectralScannerProfile",
    "apply_spectral_scanner",
    "construct_primary_metamer_pair",
    "synthetic_profile_from_contract",
]
