"""Generic cross-generation B&W density-amplitude hypothesis profile."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

PROFILE_SCHEMA = "neuro-film.bw-hybrid-density-amplitude-profile.v1"


def _identity(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class BWHybridDensityAmplitudeProfile:
    hypothesis_identity: str
    current_scalar_profile_id: str
    historical_amplitude_compiler_id: str
    densities: tuple[float, ...]
    density_rms: tuple[float, ...]
    anchor_density: float
    anchor_density_rms: float
    aperture_diameter_micrometres: float = 48.0
    current_400tx_calibration_claimed: bool = False
    historical_5233_calibration_claimed: bool = False
    scanner_response_included: bool = False

    def __post_init__(self) -> None:
        density = np.asarray(self.densities, dtype=np.float64)
        rms = np.asarray(self.density_rms, dtype=np.float64)
        if (
            self.hypothesis_identity
            != "generic_bw_cross_generation_amplitude_hypothesis"
            or not _identity(self.current_scalar_profile_id)
            or not _identity(self.historical_amplitude_compiler_id)
            or density.ndim != 1
            or density.size < 2
            or rms.shape != density.shape
            or not np.all(np.isfinite(density))
            or not np.all(np.diff(density) > 0.0)
            or not np.all(np.isfinite(rms))
            or not np.all(rms > 0.0)
            or self.anchor_density not in self.densities
            or self.aperture_diameter_micrometres != 48.0
            or self.current_400tx_calibration_claimed
            or self.historical_5233_calibration_claimed
            or self.scanner_response_included
        ):
            raise ValueError("invalid hybrid density-amplitude profile")
        anchor_index = self.densities.index(self.anchor_density)
        if self.density_rms[anchor_index] != self.anchor_density_rms:
            raise ValueError("hybrid amplitude anchor does not recover exactly")

    def sigma_d(self, density: np.ndarray | float) -> np.ndarray:
        requested = np.asarray(density, dtype=np.float64)
        nodes = np.asarray(self.densities, dtype=np.float64)
        if (
            not np.all(np.isfinite(requested))
            or np.any(requested < nodes[0])
            or np.any(requested > nodes[-1])
        ):
            raise ValueError("density is outside the hybrid source interval")
        source = np.asarray(self.density_rms, dtype=np.float64)
        result = np.exp(np.interp(requested, nodes, np.log(source)))
        for node, value in zip(nodes, source, strict=True):
            result = np.where(requested == node, value, result)
        for index in range(len(nodes) - 1):
            if source[index] == source[index + 1]:
                result = np.where(
                    (requested >= nodes[index]) & (requested <= nodes[index + 1]),
                    source[index],
                    result,
                )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "hypothesis_identity": self.hypothesis_identity,
            "current_scalar_profile_id": self.current_scalar_profile_id,
            "historical_amplitude_compiler_id": self.historical_amplitude_compiler_id,
            "densities": list(self.densities),
            "density_rms": list(self.density_rms),
            "anchor_density": self.anchor_density,
            "anchor_density_rms": self.anchor_density_rms,
            "aperture_diameter_micrometres": self.aperture_diameter_micrometres,
            "interpolation": "piecewise_log_linear_sigma_d_with_exact_nodes",
            "density_extrapolation_allowed": False,
            "current_400tx_calibration_claimed": self.current_400tx_calibration_claimed,
            "historical_5233_calibration_claimed": self.historical_5233_calibration_claimed,
            "scanner_response_included": self.scanner_response_included,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BWHybridDensityAmplitudeProfile:
        if (
            payload.get("schema") != PROFILE_SCHEMA
            or payload.get("interpolation")
            != "piecewise_log_linear_sigma_d_with_exact_nodes"
            or payload.get("density_extrapolation_allowed") is not False
        ):
            raise ValueError("unsupported hybrid density-amplitude schema")
        return cls(
            hypothesis_identity=str(payload["hypothesis_identity"]),
            current_scalar_profile_id=str(payload["current_scalar_profile_id"]),
            historical_amplitude_compiler_id=str(
                payload["historical_amplitude_compiler_id"]
            ),
            densities=tuple(float(value) for value in payload["densities"]),
            density_rms=tuple(float(value) for value in payload["density_rms"]),
            anchor_density=float(payload["anchor_density"]),
            anchor_density_rms=float(payload["anchor_density_rms"]),
            aperture_diameter_micrometres=float(
                payload["aperture_diameter_micrometres"]
            ),
            current_400tx_calibration_claimed=bool(
                payload["current_400tx_calibration_claimed"]
            ),
            historical_5233_calibration_claimed=bool(
                payload["historical_5233_calibration_claimed"]
            ),
            scanner_response_included=bool(payload["scanner_response_included"]),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = ["PROFILE_SCHEMA", "BWHybridDensityAmplitudeProfile"]
