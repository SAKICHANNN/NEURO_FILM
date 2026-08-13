"""Typed, no-extrapolation black-and-white process response profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

PROFILE_SCHEMA = "neuro-film.bw-process-response-profile.v1"


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 1 or len(result) < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with at least two values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BWProcessResponseProfile:
    film_stock_id: str
    developer_id: str
    development_time_minutes: np.ndarray
    diffuse_visual_contrast_index: np.ndarray
    source_evidence_sha256: str
    process_context: dict[str, Any]

    def __post_init__(self) -> None:
        times = _vector(self.development_time_minutes, "development times")
        contrast = _vector(self.diffuse_visual_contrast_index, "contrast indices")
        if times.shape != contrast.shape:
            raise ValueError("process-response knot arrays must have equal shape")
        if not np.all(np.diff(times) > 0.0) or not np.all(np.diff(contrast) > 0.0):
            raise ValueError("process-response knots must be strictly increasing")
        if not self.film_stock_id or not self.developer_id:
            raise ValueError("film stock and developer identities are required")
        if len(self.source_evidence_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.source_evidence_sha256
        ):
            raise ValueError("source evidence identity must be lowercase SHA-256")
        context = dict(self.process_context)
        if context != {
            "equipment": "small_tank",
            "temperature_c": 20.0,
            "densitometry": "diffuse_visual",
        }:
            raise ValueError("unsupported B&W process context")
        object.__setattr__(self, "development_time_minutes", times)
        object.__setattr__(self, "diffuse_visual_contrast_index", contrast)
        object.__setattr__(self, "process_context", MappingProxyType(context))

    def contrast_index(self, development_time_minutes: np.ndarray) -> np.ndarray:
        values = np.asarray(development_time_minutes, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("development time must be finite")
        if np.any(values < self.development_time_minutes[0]) or np.any(
            values > self.development_time_minutes[-1]
        ):
            raise ValueError("development time is outside the observed domain")
        return np.interp(
            values, self.development_time_minutes, self.diffuse_visual_contrast_index
        )

    def development_time(self, contrast_index: np.ndarray) -> np.ndarray:
        values = np.asarray(contrast_index, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("contrast index must be finite")
        if np.any(values < self.diffuse_visual_contrast_index[0]) or np.any(
            values > self.diffuse_visual_contrast_index[-1]
        ):
            raise ValueError("contrast index is outside the observed domain")
        return np.interp(
            values, self.diffuse_visual_contrast_index, self.development_time_minutes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "film_stock_id": self.film_stock_id,
            "developer_id": self.developer_id,
            "input_domain": "development_time_minutes",
            "output_domain": "diffuse_visual_contrast_index",
            "process_context": dict(self.process_context),
            "development_time_minutes": self.development_time_minutes.tolist(),
            "diffuse_visual_contrast_index": self.diffuse_visual_contrast_index.tolist(),
            "source_evidence_sha256": self.source_evidence_sha256,
            "interpolation": "piecewise_linear",
            "outside_observed_domain": "reject",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BWProcessResponseProfile:
        if (
            payload.get("schema") != PROFILE_SCHEMA
            or payload.get("input_domain") != "development_time_minutes"
            or payload.get("output_domain") != "diffuse_visual_contrast_index"
            or payload.get("interpolation") != "piecewise_linear"
            or payload.get("outside_observed_domain") != "reject"
        ):
            raise ValueError("unsupported B&W process-response profile schema")
        return cls(
            str(payload["film_stock_id"]),
            str(payload["developer_id"]),
            np.asarray(payload["development_time_minutes"], dtype=np.float64),
            np.asarray(payload["diffuse_visual_contrast_index"], dtype=np.float64),
            str(payload["source_evidence_sha256"]),
            dict(payload["process_context"]),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

