"""Typed scanner-code to optical-density OECF research primitive."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA = "neuro_film.log_scanner_oecf_profile.v1"
CALIBRATION_MODE = "development-only-held-step-checked"

_FIELDS = {
    "schema",
    "profile_id",
    "black_code",
    "gamma",
    "maximum_code",
    "minimum_calibrated_density",
    "maximum_calibrated_density",
    "source_positive_wedge_sha256",
    "source_density_table_sha256",
    "parent_evidence_sha256",
    "calibration_mode",
    "independent_confirmation",
    "production_eligible",
}


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class LogScannerOecfProfile:
    """One exact-workflow scanner OECF; it carries no product authority."""

    profile_id: str
    black_code: float
    gamma: float
    maximum_code: float
    minimum_calibrated_density: float
    maximum_calibrated_density: float
    source_positive_wedge_sha256: str
    source_density_table_sha256: str
    parent_evidence_sha256: str
    schema: str = SCHEMA
    calibration_mode: str = CALIBRATION_MODE
    independent_confirmation: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("profile_id must be a non-empty string")
        if self.schema != SCHEMA:
            raise ValueError("unsupported scanner OECF schema")
        if self.calibration_mode != CALIBRATION_MODE:
            raise ValueError("unsupported scanner OECF calibration mode")
        if self.independent_confirmation is not False:
            raise ValueError("independent confirmation is not established")
        if self.production_eligible is not False:
            raise ValueError("scanner OECF has no product authority")
        black = _finite_float(self.black_code, "black_code")
        gamma = _finite_float(self.gamma, "gamma")
        maximum = _finite_float(self.maximum_code, "maximum_code")
        minimum_density = _finite_float(
            self.minimum_calibrated_density, "minimum_calibrated_density"
        )
        maximum_density = _finite_float(
            self.maximum_calibrated_density, "maximum_calibrated_density"
        )
        if black < 0.0 or maximum <= black or gamma <= 0.0:
            raise ValueError("scanner OECF code bounds and gamma are invalid")
        if minimum_density < 0.0 or maximum_density <= minimum_density:
            raise ValueError("scanner OECF density bounds are invalid")
        _sha256(self.source_positive_wedge_sha256, "positive wedge SHA-256")
        _sha256(self.source_density_table_sha256, "density table SHA-256")
        _sha256(self.parent_evidence_sha256, "parent evidence SHA-256")

    def code_to_density(self, code: Any) -> np.ndarray:
        values = np.asarray(code, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("scanner codes must be finite")
        if np.any(values <= self.black_code) or np.any(values > self.maximum_code):
            raise ValueError("scanner code is outside the logarithmic OECF domain")
        density = self.gamma * np.log10(
            self.maximum_code / (values - self.black_code)
        )
        if not np.all(np.isfinite(density)):
            raise RuntimeError("scanner OECF produced non-finite density")
        return density

    def density_to_code(self, density: Any) -> np.ndarray:
        values = np.asarray(density, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("optical densities must be finite")
        if np.any(values < self.minimum_calibrated_density) or np.any(
            values > self.maximum_calibrated_density
        ):
            raise ValueError("optical density is outside the calibrated domain")
        code = self.black_code + self.maximum_code * np.power(
            10.0, -values / self.gamma
        )
        if np.any(code <= self.black_code) or np.any(code > self.maximum_code):
            raise RuntimeError("inverse scanner OECF left the valid code domain")
        return code

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "black_code": self.black_code,
            "gamma": self.gamma,
            "maximum_code": self.maximum_code,
            "minimum_calibrated_density": self.minimum_calibrated_density,
            "maximum_calibrated_density": self.maximum_calibrated_density,
            "source_positive_wedge_sha256": self.source_positive_wedge_sha256,
            "source_density_table_sha256": self.source_density_table_sha256,
            "parent_evidence_sha256": self.parent_evidence_sha256,
            "calibration_mode": self.calibration_mode,
            "independent_confirmation": self.independent_confirmation,
            "production_eligible": self.production_eligible,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    @property
    def profile_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LogScannerOecfProfile:
        if not isinstance(payload, dict) or set(payload) != _FIELDS:
            raise ValueError("scanner OECF profile fields must match v1 exactly")
        for label in ("independent_confirmation", "production_eligible"):
            if not isinstance(payload[label], bool):
                raise TypeError(f"{label} must be boolean")
        return cls(
            profile_id=payload["profile_id"],
            black_code=_finite_float(payload["black_code"], "black_code"),
            gamma=_finite_float(payload["gamma"], "gamma"),
            maximum_code=_finite_float(payload["maximum_code"], "maximum_code"),
            minimum_calibrated_density=_finite_float(
                payload["minimum_calibrated_density"],
                "minimum_calibrated_density",
            ),
            maximum_calibrated_density=_finite_float(
                payload["maximum_calibrated_density"],
                "maximum_calibrated_density",
            ),
            source_positive_wedge_sha256=payload["source_positive_wedge_sha256"],
            source_density_table_sha256=payload["source_density_table_sha256"],
            parent_evidence_sha256=payload["parent_evidence_sha256"],
            schema=payload["schema"],
            calibration_mode=payload["calibration_mode"],
            independent_confirmation=payload["independent_confirmation"],
            production_eligible=payload["production_eligible"],
        )


__all__ = ["CALIBRATION_MODE", "SCHEMA", "LogScannerOecfProfile"]
