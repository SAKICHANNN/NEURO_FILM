"""Typed historical B&W grain-spectrum profiles from measured source data.

The profiles in this module reproduce a published transmittance-pattern power
spectrum.  They are not modern stock defaults and are not raster renderers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import j1

PROFILE_SCHEMA = "neuro_film.historical_bw_noise_spectrum_profile.v1"


def _as_float_array(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("spatial frequencies must be finite")
    return array


@dataclass(frozen=True)
class HistoricalBWNoiseSpectrumProfile:
    """One exact density state from Ooue's measured Fuji B&W spectra."""

    source_evidence_id: str
    material_id: str
    material: str
    developer: str
    temperature_celsius: float
    development_minutes: float
    diffuse_density: float
    k: tuple[float, ...]
    p_lines_per_mm: tuple[float, ...]
    scanning_aperture_diameter_millimetres: float = 0.001
    maximum_intrinsic_frequency_lines_per_mm: float = 500.0

    def __post_init__(self) -> None:
        numeric = (
            self.temperature_celsius,
            self.development_minutes,
            self.diffuse_density,
            self.scanning_aperture_diameter_millimetres,
            self.maximum_intrinsic_frequency_lines_per_mm,
            *self.k,
            *self.p_lines_per_mm,
        )
        if (
            len(self.source_evidence_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_evidence_id
            )
            or not self.material_id
            or not self.material
            or not self.developer
            or len(self.k) not in (2, 3)
            or len(self.k) != len(self.p_lines_per_mm)
            or any(not math.isfinite(value) or value <= 0.0 for value in numeric)
            or self.scanning_aperture_diameter_millimetres != 0.001
            or self.maximum_intrinsic_frequency_lines_per_mm != 500.0
        ):
            raise ValueError("invalid historical B&W noise-spectrum profile")

    def one_dimensional(self, frequency_lines_per_mm: np.ndarray | float) -> np.ndarray:
        frequency = _as_float_array(frequency_lines_per_mm)
        output = np.zeros_like(frequency)
        for amplitude, scale in zip(self.k, self.p_lines_per_mm, strict=True):
            output += amplitude * np.exp(
                -np.square(frequency) / (2.0 * scale * scale)
            )
        return output

    def aperture_convolved_2d(
        self,
        u_lines_per_mm: np.ndarray | float,
        v_lines_per_mm: np.ndarray | float,
    ) -> np.ndarray:
        u = _as_float_array(u_lines_per_mm)
        v = _as_float_array(v_lines_per_mm)
        u, v = np.broadcast_arrays(u, v)
        radius_squared = np.square(u) + np.square(v)
        output = np.zeros_like(radius_squared)
        for amplitude, scale in zip(self.k, self.p_lines_per_mm, strict=True):
            output += (amplitude / scale) * np.exp(
                -radius_squared / (2.0 * scale * scale)
            )
        return output / math.sqrt(2.0 * math.pi)

    def circular_aperture_mtf(
        self, radial_frequency_lines_per_mm: np.ndarray | float
    ) -> np.ndarray:
        radius = _as_float_array(radial_frequency_lines_per_mm)
        if np.any(radius < 0.0):
            raise ValueError("radial frequency must be nonnegative")
        argument = (
            math.pi * self.scanning_aperture_diameter_millimetres * radius
        )
        output = np.ones_like(argument)
        nonzero = argument != 0.0
        output[nonzero] = 2.0 * j1(argument[nonzero]) / argument[nonzero]
        return output

    def intrinsic_2d(
        self,
        u_lines_per_mm: np.ndarray | float,
        v_lines_per_mm: np.ndarray | float,
    ) -> np.ndarray:
        u = _as_float_array(u_lines_per_mm)
        v = _as_float_array(v_lines_per_mm)
        u, v = np.broadcast_arrays(u, v)
        radius = np.hypot(u, v)
        if np.any(radius > self.maximum_intrinsic_frequency_lines_per_mm):
            raise ValueError("intrinsic spectrum requested outside the frozen interval")
        mtf = self.circular_aperture_mtf(radius)
        if np.any(mtf <= 0.0):
            raise ValueError("aperture MTF is not zero-free on the requested interval")
        return self.aperture_convolved_2d(u, v) / np.square(mtf)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "source_evidence_id": self.source_evidence_id,
            "material_id": self.material_id,
            "material": self.material,
            "developer": self.developer,
            "temperature_celsius": self.temperature_celsius,
            "development_minutes": self.development_minutes,
            "diffuse_density": self.diffuse_density,
            "k": list(self.k),
            "p_lines_per_mm": list(self.p_lines_per_mm),
            "spatial_frequency_unit": "lines_per_millimetre_in_grain_plane",
            "spectrum_value_semantics": (
                "transmittance_pattern_power_spectrum_under_source_normalization"
            ),
            "scanning_aperture_shape": "circular",
            "scanning_aperture_diameter_millimetres": (
                self.scanning_aperture_diameter_millimetres
            ),
            "maximum_intrinsic_frequency_lines_per_mm": (
                self.maximum_intrinsic_frequency_lines_per_mm
            ),
            "density_interpolation_allowed": False,
            "density_extrapolation_allowed": False,
            "render_allowed": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HistoricalBWNoiseSpectrumProfile:
        if (
            payload.get("schema") != PROFILE_SCHEMA
            or payload.get("spatial_frequency_unit")
            != "lines_per_millimetre_in_grain_plane"
            or payload.get("spectrum_value_semantics")
            != "transmittance_pattern_power_spectrum_under_source_normalization"
            or payload.get("scanning_aperture_shape") != "circular"
            or payload.get("density_interpolation_allowed") is not False
            or payload.get("density_extrapolation_allowed") is not False
            or payload.get("render_allowed") is not False
        ):
            raise ValueError("unsupported historical B&W noise-spectrum schema")
        return cls(
            source_evidence_id=str(payload["source_evidence_id"]),
            material_id=str(payload["material_id"]),
            material=str(payload["material"]),
            developer=str(payload["developer"]),
            temperature_celsius=float(payload["temperature_celsius"]),
            development_minutes=float(payload["development_minutes"]),
            diffuse_density=float(payload["diffuse_density"]),
            k=tuple(float(value) for value in payload["k"]),
            p_lines_per_mm=tuple(
                float(value) for value in payload["p_lines_per_mm"]
            ),
            scanning_aperture_diameter_millimetres=float(
                payload["scanning_aperture_diameter_millimetres"]
            ),
            maximum_intrinsic_frequency_lines_per_mm=float(
                payload["maximum_intrinsic_frequency_lines_per_mm"]
            ),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def profile_from_source_row(
    row: dict[str, Any], source_evidence_id: str
) -> HistoricalBWNoiseSpectrumProfile:
    return HistoricalBWNoiseSpectrumProfile(
        source_evidence_id=source_evidence_id,
        material_id=str(row["material_id"]),
        material=str(row["material"]),
        developer=str(row["developer"]),
        temperature_celsius=float(row["temperature_celsius"]),
        development_minutes=float(row["development_minutes"]),
        diffuse_density=float(row["diffuse_density"]),
        k=tuple(float(value) for value in row["k"]),
        p_lines_per_mm=tuple(float(value) for value in row["p_lines_per_mm"]),
    )


__all__ = [
    "PROFILE_SCHEMA",
    "HistoricalBWNoiseSpectrumProfile",
    "profile_from_source_row",
]

