"""Bounded monotone manufacturer characteristic-curve priors.

The input is explicit relative layer log exposure, not ordinary image RGB.
The output is measured density in the source graph's densitometry domain.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

import numpy as np

CURVE_SCHEMA = "neuro_film.manufacturer_characteristic_curve.v1"
BUNDLE_SCHEMA = "neuro_film.manufacturer_characteristic_prior.v1"
CHANNELS = ("red", "green", "blue")


def _readonly_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 1 or len(result) < 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ManufacturerCharacteristicCurve:
    """One no-extrapolation piecewise-linear layer characteristic curve."""

    layer: str
    log_exposure_knots: np.ndarray
    density_knots: np.ndarray

    def __post_init__(self) -> None:
        if self.layer not in CHANNELS:
            raise ValueError("unsupported characteristic layer")
        exposure = _readonly_vector(self.log_exposure_knots, name="log exposure knots")
        density = _readonly_vector(self.density_knots, name="density knots")
        if exposure.shape != density.shape:
            raise ValueError("characteristic knot arrays must have equal shape")
        if not np.all(np.diff(exposure) > 0.0):
            raise ValueError("log exposure knots must be strictly increasing")
        if not np.all(np.diff(density) >= 0.0):
            raise ValueError("density knots must be monotone nondecreasing")
        object.__setattr__(self, "log_exposure_knots", exposure)
        object.__setattr__(self, "density_knots", density)

    @property
    def domain(self) -> tuple[float, float]:
        return float(self.log_exposure_knots[0]), float(self.log_exposure_knots[-1])

    @property
    def density_bounds(self) -> tuple[float, float]:
        return float(self.density_knots[0]), float(self.density_knots[-1])

    def apply(self, log_exposure: np.ndarray) -> np.ndarray:
        values = np.asarray(log_exposure, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("relative layer log exposure must be finite")
        lower, upper = self.domain
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("relative layer log exposure is outside the observed domain")
        return np.interp(values, self.log_exposure_knots, self.density_knots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CURVE_SCHEMA,
            "layer": self.layer,
            "log_exposure_knots": self.log_exposure_knots.tolist(),
            "density_knots": self.density_knots.tolist(),
            "interpolation": "piecewise_linear",
            "outside_observed_domain": "reject",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManufacturerCharacteristicCurve":
        if (
            payload.get("schema") != CURVE_SCHEMA
            or payload.get("interpolation") != "piecewise_linear"
            or payload.get("outside_observed_domain") != "reject"
        ):
            raise ValueError("unsupported manufacturer characteristic curve schema")
        return cls(
            str(payload["layer"]),
            np.asarray(payload["log_exposure_knots"], dtype=np.float64),
            np.asarray(payload["density_knots"], dtype=np.float64),
        )


@dataclass(frozen=True)
class ManufacturerCharacteristicPrior:
    """Three layer curves in conventional red/green/blue array order."""

    curves: tuple[
        ManufacturerCharacteristicCurve,
        ManufacturerCharacteristicCurve,
        ManufacturerCharacteristicCurve,
    ]
    source_evidence_id: str
    measurement_context: dict[str, str]
    input_domain: str = "relative_layer_log_exposure"
    output_domain: str = "status_m_density"

    def __post_init__(self) -> None:
        if (
            len(self.curves) != 3
            or not all(isinstance(curve, ManufacturerCharacteristicCurve) for curve in self.curves)
            or tuple(curve.layer for curve in self.curves) != CHANNELS
        ):
            raise ValueError("manufacturer prior curves must be ordered red, green, blue")
        if (
            not isinstance(self.source_evidence_id, str)
            or len(self.source_evidence_id) != 64
            or any(character not in "0123456789abcdef" for character in self.source_evidence_id)
        ):
            raise ValueError("source evidence identity must be lowercase SHA-256")
        if self.input_domain != "relative_layer_log_exposure":
            raise ValueError("unsupported manufacturer prior input domain")
        if self.output_domain != "status_m_density":
            raise ValueError("unsupported manufacturer prior output domain")
        context = dict(self.measurement_context)
        if context != {
            "exposure": "5500 K daylight",
            "process": "ECN-2",
            "densitometry": "Status M",
        }:
            raise ValueError("manufacturer prior measurement context mismatch")
        object.__setattr__(self, "measurement_context", MappingProxyType(context))

    def apply(self, relative_layer_log_exposure: np.ndarray) -> np.ndarray:
        values = np.asarray(relative_layer_log_exposure, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != 3:
            raise ValueError("manufacturer prior input must end in three RGB-ordered layers")
        if not np.all(np.isfinite(values)):
            raise ValueError("manufacturer prior input must be finite")
        return np.stack(
            [self.curves[index].apply(values[..., index]) for index in range(3)], axis=-1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BUNDLE_SCHEMA,
            "input_domain": self.input_domain,
            "output_domain": self.output_domain,
            "channel_order": list(CHANNELS),
            "source_evidence_id": self.source_evidence_id,
            "measurement_context": dict(self.measurement_context),
            "curves": [curve.to_dict() for curve in self.curves],
            "claim_ceiling": "manufacturer graph prior; layer exposure must be supplied explicitly",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManufacturerCharacteristicPrior":
        if (
            payload.get("schema") != BUNDLE_SCHEMA
            or payload.get("channel_order") != list(CHANNELS)
            or payload.get("claim_ceiling")
            != "manufacturer graph prior; layer exposure must be supplied explicitly"
        ):
            raise ValueError("unsupported manufacturer characteristic prior schema")
        curves = tuple(
            ManufacturerCharacteristicCurve.from_dict(item) for item in payload["curves"]
        )
        return cls(
            curves,  # type: ignore[arg-type]
            str(payload["source_evidence_id"]),
            dict(payload["measurement_context"]),
            str(payload["input_domain"]),
            str(payload["output_domain"]),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BUNDLE_SCHEMA",
    "CHANNELS",
    "CURVE_SCHEMA",
    "ManufacturerCharacteristicCurve",
    "ManufacturerCharacteristicPrior",
]
