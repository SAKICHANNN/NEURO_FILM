"""Typed aperture scaling for retained diffuse-RMS density amplitudes.

This module changes measurement aperture only.  It does not select a spatial
spectrum, synthesize pixels, or interpret compound-Poisson events as grains.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.transmittance_granularity import (
    ApertureCellCompoundPoissonParameters,
    ApertureCellCompoundPoissonProfile,
    TransmittanceGranularityMoments,
    density_compound_poisson_to_transmittance_moments,
)

PROFILE_SCHEMA = "neuro_film.aperture_scaled_compound_poisson_profile.v1"


def scale_density_rms_between_apertures(
    density_rms: np.ndarray,
    source_aperture_micrometres: float,
    target_aperture_micrometres: float,
) -> np.ndarray:
    """Apply the fixed inverse-diameter diffuse-RMS aperture law."""

    values = np.asarray(density_rms, dtype=np.float64)
    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or not math.isfinite(source_aperture_micrometres)
        or not math.isfinite(target_aperture_micrometres)
        or source_aperture_micrometres <= 0.0
        or target_aperture_micrometres <= 0.0
    ):
        raise ValueError("invalid density-RMS aperture scaling request")
    return values * (source_aperture_micrometres / target_aperture_micrometres)


@dataclass(frozen=True)
class ApertureScaledCompoundPoissonProfile:
    """Bounded multi-aperture wrapper around the retained 48um profile."""

    parent_profile: ApertureCellCompoundPoissonProfile
    source_evidence_id: str
    reference_aperture_micrometres: float = 48.0
    minimum_aperture_micrometres: float = 7.25
    maximum_aperture_micrometres: float = 384.0

    def __post_init__(self) -> None:
        if (
            len(self.source_evidence_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_evidence_id
            )
            or self.reference_aperture_micrometres != 48.0
            or self.minimum_aperture_micrometres != 7.25
            or self.maximum_aperture_micrometres != 384.0
        ):
            raise ValueError("invalid aperture-scaled profile")

    def _validate_target(self, target_aperture_micrometres: float) -> float:
        target = float(target_aperture_micrometres)
        if (
            not math.isfinite(target)
            or target < self.minimum_aperture_micrometres
            or target > self.maximum_aperture_micrometres
        ):
            raise ValueError("target aperture is outside the evidence-backed range")
        return target

    def scale_density_moments(
        self,
        density_mean: np.ndarray,
        density_rms_at_reference: np.ndarray,
        target_aperture_micrometres: float,
    ) -> tuple[ApertureCellCompoundPoissonParameters, TransmittanceGranularityMoments]:
        target = self._validate_target(target_aperture_micrometres)
        mean = np.asarray(density_mean, dtype=np.float64)
        rms = scale_density_rms_between_apertures(
            density_rms_at_reference,
            self.reference_aperture_micrometres,
            target,
        )
        return density_compound_poisson_to_transmittance_moments(mean, rms)

    def evaluate(
        self,
        prior: ManufacturerCharacteristicPrior,
        relative_layer_log_exposure: np.ndarray,
        target_aperture_micrometres: float,
    ) -> tuple[ApertureCellCompoundPoissonParameters, TransmittanceGranularityMoments]:
        _, reference = self.parent_profile.evaluate(
            prior, relative_layer_log_exposure
        )
        return self.scale_density_moments(
            reference.density_mean,
            reference.density_rms,
            target_aperture_micrometres,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "parent_profile": self.parent_profile.to_dict(),
            "source_evidence_id": self.source_evidence_id,
            "reference_aperture_micrometres": self.reference_aperture_micrometres,
            "minimum_aperture_micrometres": self.minimum_aperture_micrometres,
            "maximum_aperture_micrometres": self.maximum_aperture_micrometres,
            "density_mean_scaling": "identity",
            "density_rms_scaling": (
                "sigma_to = sigma_from * aperture_from / aperture_to"
            ),
            "fixed_exponent": 1.0,
            "spatial_nps_status": "unidentified",
            "render_allowed": False,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> ApertureScaledCompoundPoissonProfile:
        if (
            payload.get("schema") != PROFILE_SCHEMA
            or payload.get("density_mean_scaling") != "identity"
            or payload.get("density_rms_scaling")
            != "sigma_to = sigma_from * aperture_from / aperture_to"
            or payload.get("fixed_exponent") != 1.0
            or payload.get("spatial_nps_status") != "unidentified"
            or payload.get("render_allowed") is not False
        ):
            raise ValueError("unsupported aperture-scaled profile schema")
        return cls(
            ApertureCellCompoundPoissonProfile.from_dict(payload["parent_profile"]),
            str(payload["source_evidence_id"]),
            float(payload["reference_aperture_micrometres"]),
            float(payload["minimum_aperture_micrometres"]),
            float(payload["maximum_aperture_micrometres"]),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PROFILE_SCHEMA",
    "ApertureScaledCompoundPoissonProfile",
    "scale_density_rms_between_apertures",
]
