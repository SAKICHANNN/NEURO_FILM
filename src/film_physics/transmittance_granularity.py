"""Typed density-to-transmittance granularity moment conversion.

This module converts a density-domain diffuse-rms amplitude through the
physical ``T = 10**(-D)`` relation.  It deliberately contains no spatial
spectrum, grain geometry, scanner response or display-RGB noise operation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.film_physics.granularity_amplitude import GranularityAmplitudeProfile
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior

PROFILE_SCHEMA = "neuro_film.transmittance_granularity_profile.v1"
_LOG_10 = float(np.log(10.0))


@dataclass(frozen=True)
class TransmittanceGranularityMoments:
    density_mean: np.ndarray
    density_rms: np.ndarray
    transmittance_mean: np.ndarray
    transmittance_rms: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.array(value, dtype=np.float64, copy=True)
            for value in (
                self.density_mean,
                self.density_rms,
                self.transmittance_mean,
                self.transmittance_rms,
            )
        )
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("granularity moment shapes must match")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("granularity moments must be finite")
        if np.any(arrays[0] < 0.0) or any(np.any(array <= 0.0) for array in arrays[1:]):
            raise ValueError("granularity moments are outside the physical domain")
        for name, array in zip(self.__dataclass_fields__, arrays, strict=True):
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def density_gaussian_to_transmittance_moments(
    density_mean: np.ndarray, density_rms: np.ndarray
) -> TransmittanceGranularityMoments:
    """Map Gaussian density moments to exact log-normal transmittance moments."""

    mean = np.asarray(density_mean, dtype=np.float64)
    sigma = np.asarray(density_rms, dtype=np.float64)
    if (
        mean.shape != sigma.shape
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(sigma))
    ):
        raise ValueError("density moments must be finite and shape-matched")
    if np.any(mean < 0.0) or np.any(sigma <= 0.0):
        raise ValueError("density mean/rms must be positive physical values")
    log_variance = np.square(_LOG_10 * sigma)
    transmittance_mean = np.exp(-_LOG_10 * mean + 0.5 * log_variance)
    transmittance_rms = transmittance_mean * np.sqrt(np.expm1(log_variance))
    return TransmittanceGranularityMoments(
        mean, sigma, transmittance_mean, transmittance_rms
    )


def transmittance_moments_to_density_gaussian(
    transmittance_mean: np.ndarray, transmittance_rms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Invert exact log-normal transmittance moments back to density moments."""

    mean = np.asarray(transmittance_mean, dtype=np.float64)
    sigma = np.asarray(transmittance_rms, dtype=np.float64)
    if (
        mean.shape != sigma.shape
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(sigma))
    ):
        raise ValueError("transmittance moments must be finite and shape-matched")
    if np.any(mean <= 0.0) or np.any(sigma <= 0.0):
        raise ValueError("transmittance moments must be positive")
    log_variance = np.log1p(np.square(sigma / mean))
    density_rms = np.sqrt(log_variance) / _LOG_10
    density_mean = (-np.log(mean) + 0.5 * log_variance) / _LOG_10
    if np.any(density_mean < 0.0):
        raise ValueError("inverted density mean is outside the physical domain")
    return density_mean, density_rms


def density_delta_method_transmittance_rms(
    density_mean: np.ndarray, density_rms: np.ndarray
) -> np.ndarray:
    """First-order diagnostic; never used in the exact compiler output."""

    mean = np.asarray(density_mean, dtype=np.float64)
    sigma = np.asarray(density_rms, dtype=np.float64)
    if mean.shape != sigma.shape or np.any(mean < 0.0) or np.any(sigma <= 0.0):
        raise ValueError("invalid density moments")
    return _LOG_10 * np.power(10.0, -mean) * sigma


@dataclass(frozen=True)
class TransmittanceGranularityProfile:
    density_amplitude_profile: GranularityAmplitudeProfile
    source_evidence_id: str
    distribution_assumption: str = "gaussian_small_fluctuation_compatibility"

    def __post_init__(self) -> None:
        if (
            len(self.source_evidence_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_evidence_id
            )
            or self.distribution_assumption
            != "gaussian_small_fluctuation_compatibility"
        ):
            raise ValueError("invalid transmittance granularity profile")

    def evaluate(
        self,
        prior: ManufacturerCharacteristicPrior,
        relative_layer_log_exposure: np.ndarray,
    ) -> TransmittanceGranularityMoments:
        values = np.asarray(relative_layer_log_exposure, dtype=np.float64)
        density_mean = prior.apply(values)
        density_rms = self.density_amplitude_profile.evaluate(prior, values)
        return density_gaussian_to_transmittance_moments(density_mean, density_rms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "input_domain": "relative_layer_log_exposure",
            "intermediate_domain": "status_m_density_mean_and_diffuse_rms_at_48um",
            "output_domain": "transmittance_mean_and_rms_at_48um",
            "transform": "transmittance = pow(10, -density)",
            "moment_family": "exact_log_normal",
            "distribution_assumption": self.distribution_assumption,
            "source_evidence_id": self.source_evidence_id,
            "density_amplitude_profile": self.density_amplitude_profile.to_dict(),
            "spatial_structure_status": "unidentified",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransmittanceGranularityProfile:
        if (
            payload.get("schema") != PROFILE_SCHEMA
            or payload.get("input_domain") != "relative_layer_log_exposure"
            or payload.get("intermediate_domain")
            != "status_m_density_mean_and_diffuse_rms_at_48um"
            or payload.get("output_domain") != "transmittance_mean_and_rms_at_48um"
            or payload.get("transform") != "transmittance = pow(10, -density)"
            or payload.get("moment_family") != "exact_log_normal"
            or payload.get("spatial_structure_status") != "unidentified"
        ):
            raise ValueError("unsupported transmittance granularity profile schema")
        return cls(
            GranularityAmplitudeProfile.from_dict(payload["density_amplitude_profile"]),
            str(payload["source_evidence_id"]),
            str(payload["distribution_assumption"]),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PROFILE_SCHEMA",
    "TransmittanceGranularityMoments",
    "TransmittanceGranularityProfile",
    "density_delta_method_transmittance_rms",
    "density_gaussian_to_transmittance_moments",
    "transmittance_moments_to_density_gaussian",
]
