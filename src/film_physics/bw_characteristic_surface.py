"""Typed no-extrapolation B&W development-time characteristic surface."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

SURFACE_SCHEMA = "neuro-film.bw-characteristic-surface.v1"


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 1 or len(result) < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector")
    if not np.all(np.diff(result) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BWCharacteristicSurface:
    film_stock_id: str
    developer_id: str
    development_time_minutes: np.ndarray
    log_exposure_knots: np.ndarray
    diffuse_visual_density: np.ndarray
    source_evidence_sha256: str
    process_context: dict[str, Any]

    def __post_init__(self) -> None:
        times = _vector(self.development_time_minutes, "development times")
        exposure = _vector(self.log_exposure_knots, "log exposure knots")
        density = np.array(self.diffuse_visual_density, dtype=np.float64, copy=True)
        if density.shape != (len(times), len(exposure)) or not np.all(
            np.isfinite(density)
        ):
            raise ValueError("density table shape or values are invalid")
        if not np.all(np.diff(density, axis=1) > 0.0):
            raise ValueError("each characteristic row must increase with exposure")
        if not np.all(np.diff(density, axis=0) > 0.0):
            raise ValueError("density must increase with development time")
        if not self.film_stock_id or not self.developer_id:
            raise ValueError("film stock and developer identities are required")
        if len(self.source_evidence_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.source_evidence_sha256
        ):
            raise ValueError("source evidence identity must be lowercase SHA-256")
        context = dict(self.process_context)
        if context != {
            "agitation": "one_minute_intervals",
            "densitometry": "diffuse_visual",
            "equipment": "large_tank",
            "temperature_c": 20.0,
        }:
            raise ValueError("unsupported B&W characteristic process context")
        density.setflags(write=False)
        object.__setattr__(self, "development_time_minutes", times)
        object.__setattr__(self, "log_exposure_knots", exposure)
        object.__setattr__(self, "diffuse_visual_density", density)
        object.__setattr__(self, "process_context", MappingProxyType(context))

    def _validate_time(self, value: float) -> float:
        time = float(value)
        if (
            not np.isfinite(time)
            or not self.development_time_minutes[0]
            <= time
            <= self.development_time_minutes[-1]
        ):
            raise ValueError("development time is outside the observed domain")
        return time

    def _density_row(self, time: float) -> np.ndarray:
        upper = int(np.searchsorted(self.development_time_minutes, time, side="right"))
        if upper == 0:
            return self.diffuse_visual_density[0]
        if upper == len(self.development_time_minutes):
            return self.diffuse_visual_density[-1]
        lower = upper - 1
        weight = (time - self.development_time_minutes[lower]) / (
            self.development_time_minutes[upper] - self.development_time_minutes[lower]
        )
        return (1.0 - weight) * self.diffuse_visual_density[
            lower
        ] + weight * self.diffuse_visual_density[upper]

    def density(
        self, development_time_minutes: float, log_exposure: np.ndarray
    ) -> np.ndarray:
        time = self._validate_time(development_time_minutes)
        values = np.asarray(log_exposure, dtype=np.float64)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < self.log_exposure_knots[0])
            or np.any(values > self.log_exposure_knots[-1])
        ):
            raise ValueError("log exposure is outside the observed domain")
        return np.interp(values, self.log_exposure_knots, self._density_row(time))

    def log_exposure(
        self, development_time_minutes: float, density: np.ndarray
    ) -> np.ndarray:
        time = self._validate_time(development_time_minutes)
        values = np.asarray(density, dtype=np.float64)
        row = self._density_row(time)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < row[0])
            or np.any(values > row[-1])
        ):
            raise ValueError("density is outside the observed domain")
        return np.interp(values, row, self.log_exposure_knots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SURFACE_SCHEMA,
            "film_stock_id": self.film_stock_id,
            "developer_id": self.developer_id,
            "input_domains": ["development_time_minutes", "relative_log_exposure"],
            "output_domain": "diffuse_visual_density",
            "development_time_minutes": self.development_time_minutes.tolist(),
            "log_exposure_knots": self.log_exposure_knots.tolist(),
            "diffuse_visual_density": self.diffuse_visual_density.tolist(),
            "source_evidence_sha256": self.source_evidence_sha256,
            "process_context": dict(self.process_context),
            "interpolation": "bilinear_piecewise_linear",
            "outside_observed_domain": "reject",
            "claim_ceiling": "source-derived B&W characteristic interpolation; unresolved shape physics",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BWCharacteristicSurface:
        if (
            payload.get("schema") != SURFACE_SCHEMA
            or payload.get("input_domains")
            != ["development_time_minutes", "relative_log_exposure"]
            or payload.get("output_domain") != "diffuse_visual_density"
            or payload.get("interpolation") != "bilinear_piecewise_linear"
            or payload.get("outside_observed_domain") != "reject"
            or payload.get("claim_ceiling")
            != "source-derived B&W characteristic interpolation; unresolved shape physics"
        ):
            raise ValueError("unsupported B&W characteristic surface schema")
        return cls(
            str(payload["film_stock_id"]),
            str(payload["developer_id"]),
            np.asarray(payload["development_time_minutes"], dtype=np.float64),
            np.asarray(payload["log_exposure_knots"], dtype=np.float64),
            np.asarray(payload["diffuse_visual_density"], dtype=np.float64),
            str(payload["source_evidence_sha256"]),
            dict(payload["process_context"]),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["SURFACE_SCHEMA", "BWCharacteristicSurface"]
