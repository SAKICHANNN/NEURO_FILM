"""Source-bounded relative amplitude compiler for historical B&W Wiener data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

PROFILE_SCHEMA = "neuro-film.bw-wiener-amplitude-compiler.v1"


@dataclass(frozen=True)
class BWWienerAmplitudeCompiler:
    source_profile_id: str
    densities: tuple[float, ...]
    wiener_granularity_spectrum_cm2: tuple[float, ...]
    reference_density: float
    interpolation: str = "piecewise_log_linear_wiener_then_sqrt_relative_amplitude"
    current_400tx_claimed: bool = False
    spatial_spectrum_claimed: bool = False
    render_allowed: bool = False

    def __post_init__(self) -> None:
        density = np.asarray(self.densities, dtype=np.float64)
        wiener = np.asarray(self.wiener_granularity_spectrum_cm2, dtype=np.float64)
        if (
            len(self.source_profile_id) != 64
            or density.ndim != 1
            or density.size < 2
            or wiener.shape != density.shape
            or not np.all(np.isfinite(density))
            or not np.all(np.diff(density) > 0.0)
            or not np.all(np.isfinite(wiener))
            or not np.all(wiener > 0.0)
            or not density[0] <= self.reference_density <= density[-1]
            or self.interpolation
            != "piecewise_log_linear_wiener_then_sqrt_relative_amplitude"
            or self.current_400tx_claimed
            or self.spatial_spectrum_claimed
            or self.render_allowed
        ):
            raise ValueError("invalid historical Wiener amplitude compiler")

    def relative_standard_deviation(self, density: np.ndarray | float) -> np.ndarray:
        requested = np.asarray(density, dtype=np.float64)
        nodes = np.asarray(self.densities, dtype=np.float64)
        if (
            not np.all(np.isfinite(requested))
            or np.any(requested < nodes[0])
            or np.any(requested > nodes[-1])
        ):
            raise ValueError("density is outside the historical source interval")
        log_wiener = np.log(
            np.asarray(self.wiener_granularity_spectrum_cm2, dtype=np.float64)
        )
        source_wiener = np.asarray(
            self.wiener_granularity_spectrum_cm2, dtype=np.float64
        )
        interpolated_wiener = np.exp(np.interp(requested, nodes, log_wiener))
        for node, value in zip(nodes, source_wiener, strict=True):
            interpolated_wiener = np.where(
                requested == node, value, interpolated_wiener
            )
        for index in range(len(nodes) - 1):
            if source_wiener[index] == source_wiener[index + 1]:
                plateau = (requested >= nodes[index]) & (requested <= nodes[index + 1])
                interpolated_wiener = np.where(
                    plateau, source_wiener[index], interpolated_wiener
                )
        reference_wiener = float(
            np.exp(np.interp(self.reference_density, nodes, log_wiener))
        )
        for node, value in zip(nodes, source_wiener, strict=True):
            if self.reference_density == node:
                reference_wiener = float(value)
                break
        return np.sqrt(interpolated_wiener / reference_wiener)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "source_profile_id": self.source_profile_id,
            "densities": list(self.densities),
            "wiener_granularity_spectrum_cm2": list(
                self.wiener_granularity_spectrum_cm2
            ),
            "reference_density": self.reference_density,
            "interpolation": self.interpolation,
            "density_extrapolation_allowed": False,
            "current_400tx_claimed": self.current_400tx_claimed,
            "spatial_spectrum_claimed": self.spatial_spectrum_claimed,
            "render_allowed": self.render_allowed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BWWienerAmplitudeCompiler:
        if (
            payload.get("schema") != PROFILE_SCHEMA
            or payload.get("density_extrapolation_allowed") is not False
        ):
            raise ValueError("unsupported Wiener amplitude compiler schema")
        return cls(
            source_profile_id=str(payload["source_profile_id"]),
            densities=tuple(float(value) for value in payload["densities"]),
            wiener_granularity_spectrum_cm2=tuple(
                float(value) for value in payload["wiener_granularity_spectrum_cm2"]
            ),
            reference_density=float(payload["reference_density"]),
            interpolation=str(payload["interpolation"]),
            current_400tx_claimed=bool(payload["current_400tx_claimed"]),
            spatial_spectrum_claimed=bool(payload["spatial_spectrum_claimed"]),
            render_allowed=bool(payload["render_allowed"]),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = ["PROFILE_SCHEMA", "BWWienerAmplitudeCompiler"]
