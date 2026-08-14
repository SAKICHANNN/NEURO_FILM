"""Typed single-observation B&W diffuse-rms granularity amplitude."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

PROFILE_SCHEMA = "neuro-film.bw-granularity-scalar-profile.v1"


@dataclass(frozen=True)
class BWGranularityScalarProfile:
    film_stock_id: str
    density_mean: float
    density_rms: float
    aperture_diameter_micrometres: float
    densitometry: str
    source_evidence_id: str
    spatial_nps_status: str = "unidentified"
    render_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.film_stock_id:
            raise ValueError("film stock identity is required")
        numeric = (
            self.density_mean,
            self.density_rms,
            self.aperture_diameter_micrometres,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("granularity observation values must be positive")
        if self.densitometry != "diffuse_visual":
            raise ValueError("unsupported densitometry")
        if len(self.source_evidence_id) != 64 or any(
            value not in "0123456789abcdef" for value in self.source_evidence_id
        ):
            raise ValueError("source evidence identity must be lowercase SHA-256")
        if (
            self.spatial_nps_status != "unidentified"
            or self.render_allowed is not False
        ):
            raise ValueError("scalar granularity does not authorize spatial rendering")

    def density_rms_at(
        self, *, density_mean: float, aperture_diameter_micrometres: float
    ) -> float:
        density = float(density_mean)
        aperture = float(aperture_diameter_micrometres)
        if density != self.density_mean:
            raise ValueError("density is outside the single observed point")
        if aperture != self.aperture_diameter_micrometres:
            raise ValueError("aperture is outside the single observed point")
        return self.density_rms

    def render(self, *_args: Any, **_kwargs: Any) -> None:
        raise ValueError("scalar granularity profile cannot render spatial structure")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "film_stock_id": self.film_stock_id,
            "density_mean": self.density_mean,
            "density_rms": self.density_rms,
            "aperture_diameter_micrometres": self.aperture_diameter_micrometres,
            "densitometry": self.densitometry,
            "source_evidence_id": self.source_evidence_id,
            "spatial_nps_status": self.spatial_nps_status,
            "render_allowed": self.render_allowed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BWGranularityScalarProfile:
        if payload.get("schema") != PROFILE_SCHEMA:
            raise ValueError("unsupported B&W granularity scalar schema")
        return cls(
            str(payload["film_stock_id"]),
            float(payload["density_mean"]),
            float(payload["density_rms"]),
            float(payload["aperture_diameter_micrometres"]),
            str(payload["densitometry"]),
            str(payload["source_evidence_id"]),
            str(payload["spatial_nps_status"]),
            payload["render_allowed"],
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["PROFILE_SCHEMA", "BWGranularityScalarProfile"]
