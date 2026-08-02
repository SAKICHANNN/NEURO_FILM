"""Typed diffuse-rms granularity amplitude profiles.

The profile describes measured density variance at one stated aperture. It does
not describe a spatial spectrum or a display-RGB noise layer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

PROFILE_SCHEMA = "neuro_film.granularity_amplitude_profile.v1"
CHANNELS = ("red", "green", "blue")


def _valid_identity(value: str) -> bool:
    return (
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class GranularityAmplitudeProfile:
    """One bounded characteristic-slope density-variance profile."""

    characteristic_prior_identity: str
    source_evidence_id: str
    channel_floor_variance: dict[str, float]
    shared_amplitude: float
    aperture_diameter_micrometres: float = 48.0
    input_domain: str = "relative_layer_log_exposure"
    output_domain: str = "diffuse_rms_status_m_density_at_stated_aperture"

    def __post_init__(self) -> None:
        if not _valid_identity(self.characteristic_prior_identity) or not _valid_identity(
            self.source_evidence_id
        ):
            raise ValueError("granularity profile identities must be lowercase SHA-256")
        floors = dict(self.channel_floor_variance)
        if tuple(floors) != CHANNELS or any(
            not np.isfinite(value) or value <= 0.0 for value in floors.values()
        ):
            raise ValueError("granularity variance floors must be positive RGB values")
        if not np.isfinite(self.shared_amplitude) or self.shared_amplitude <= 0.0:
            raise ValueError("granularity shared amplitude must be positive")
        if self.aperture_diameter_micrometres != 48.0:
            raise ValueError("unsupported granularity measurement aperture")
        if self.input_domain != "relative_layer_log_exposure":
            raise ValueError("unsupported granularity input domain")
        if self.output_domain != "diffuse_rms_status_m_density_at_stated_aperture":
            raise ValueError("unsupported granularity output domain")
        object.__setattr__(self, "channel_floor_variance", MappingProxyType(floors))

    def _validate_prior(self, prior: ManufacturerCharacteristicPrior) -> None:
        if prior.identity() != self.characteristic_prior_identity:
            raise ValueError("granularity characteristic prior identity mismatch")

    @staticmethod
    def _slope(curve: Any, exposure: np.ndarray) -> np.ndarray:
        knots = curve.log_exposure_knots
        if np.any(exposure < knots[0]) or np.any(exposure > knots[-1]):
            raise ValueError("granularity exposure is outside the observed domain")
        indices = np.searchsorted(knots, exposure, side="right") - 1
        indices = np.clip(indices, 0, len(knots) - 2)
        return (curve.density_knots[indices + 1] - curve.density_knots[indices]) / (
            knots[indices + 1] - knots[indices]
        )

    def evaluate_channel(
        self,
        prior: ManufacturerCharacteristicPrior,
        channel: str,
        log_exposure: np.ndarray,
    ) -> np.ndarray:
        """Return Sigma-D at the profile's stated aperture for one layer."""

        self._validate_prior(prior)
        if channel not in CHANNELS:
            raise ValueError("unsupported granularity channel")
        values = np.asarray(log_exposure, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("granularity exposure must be finite")
        curve = prior.curves[CHANNELS.index(channel)]
        slope = self._slope(curve, values)
        variance = float(self.channel_floor_variance[channel]) + self.shared_amplitude * (
            slope * slope / np.power(10.0, values)
        )
        result = np.sqrt(variance)
        if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError("granularity amplitude is invalid")
        return result

    def evaluate(
        self,
        prior: ManufacturerCharacteristicPrior,
        relative_layer_log_exposure: np.ndarray,
    ) -> np.ndarray:
        """Return RGB-ordered Sigma-D values without spatial synthesis."""

        values = np.asarray(relative_layer_log_exposure, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != 3:
            raise ValueError("granularity input must end in three RGB layers")
        return np.stack(
            [
                self.evaluate_channel(prior, channel, values[..., index])
                for index, channel in enumerate(CHANNELS)
            ],
            axis=-1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "input_domain": self.input_domain,
            "output_domain": self.output_domain,
            "channel_order": list(CHANNELS),
            "aperture_diameter_micrometres": self.aperture_diameter_micrometres,
            "characteristic_prior_identity": self.characteristic_prior_identity,
            "source_evidence_id": self.source_evidence_id,
            "channel_floor_variance": dict(self.channel_floor_variance),
            "shared_amplitude": self.shared_amplitude,
            "variance_family": (
                "channel_floor_variance + shared_amplitude * "
                "square(piecewise_linear_density_slope_per_log10_exposure) / "
                "pow(10, log10_exposure)"
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GranularityAmplitudeProfile:
        if (
            payload.get("schema") != PROFILE_SCHEMA
            or payload.get("channel_order") != list(CHANNELS)
            or payload.get("variance_family")
            != (
                "channel_floor_variance + shared_amplitude * "
                "square(piecewise_linear_density_slope_per_log10_exposure) / "
                "pow(10, log10_exposure)"
            )
        ):
            raise ValueError("unsupported granularity amplitude profile schema")
        return cls(
            characteristic_prior_identity=str(payload["characteristic_prior_identity"]),
            source_evidence_id=str(payload["source_evidence_id"]),
            channel_floor_variance=dict(payload["channel_floor_variance"]),
            shared_amplitude=float(payload["shared_amplitude"]),
            aperture_diameter_micrometres=float(
                payload["aperture_diameter_micrometres"]
            ),
            input_domain=str(payload["input_domain"]),
            output_domain=str(payload["output_domain"]),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["CHANNELS", "PROFILE_SCHEMA", "GranularityAmplitudeProfile"]
