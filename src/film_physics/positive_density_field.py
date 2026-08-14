"""Intrinsically positive developed-density field transforms."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

PROFILE_SCHEMA = "neuro-film.softplus-density-parameter-profile.v1"


@dataclass(frozen=True)
class SoftplusDensityParameterProfile:
    source_evidence_id: str
    densities: tuple[float, ...]
    a: tuple[float, ...]
    b: tuple[float, ...]

    def __post_init__(self) -> None:
        densities = np.asarray(self.densities, dtype=np.float64)
        a = np.asarray(self.a, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)
        if (
            len(self.source_evidence_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_evidence_id
            )
            or densities.ndim != 1
            or densities.size < 2
            or a.shape != densities.shape
            or b.shape != densities.shape
            or not np.all(np.isfinite(densities))
            or not np.all(np.diff(densities) > 0.0)
            or not np.all(np.isfinite(a))
            or not np.all(np.isfinite(b))
            or not np.all(b > 0.0)
        ):
            raise ValueError("invalid softplus density-parameter profile")

    def parameters(self, density: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(density, dtype=np.float64)
        nodes = np.asarray(self.densities, dtype=np.float64)
        if (
            not np.all(np.isfinite(requested))
            or np.any(requested < nodes[0])
            or np.any(requested > nodes[-1])
        ):
            raise ValueError("density is outside the parameter profile")
        source_a = np.asarray(self.a, dtype=np.float64)
        source_b = np.asarray(self.b, dtype=np.float64)
        result_a = np.interp(requested, nodes, source_a)
        result_b = np.exp(np.interp(requested, nodes, np.log(source_b)))
        for node, value_a, value_b in zip(nodes, source_a, source_b, strict=True):
            result_a = np.where(requested == node, value_a, result_a)
            result_b = np.where(requested == node, value_b, result_b)
        return result_a, result_b

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "source_evidence_id": self.source_evidence_id,
            "densities": list(self.densities),
            "a": list(self.a),
            "b": list(self.b),
            "a_interpolation": "piecewise_linear_in_density",
            "b_interpolation": "piecewise_log_linear_in_density",
            "density_extrapolation_allowed": False,
            "parameter_refit_allowed": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SoftplusDensityParameterProfile:
        if (
            payload.get("schema") != PROFILE_SCHEMA
            or payload.get("a_interpolation") != "piecewise_linear_in_density"
            or payload.get("b_interpolation") != "piecewise_log_linear_in_density"
            or payload.get("density_extrapolation_allowed") is not False
            or payload.get("parameter_refit_allowed") is not False
        ):
            raise ValueError("unsupported softplus parameter profile schema")
        return cls(
            source_evidence_id=str(payload["source_evidence_id"]),
            densities=tuple(float(value) for value in payload["densities"]),
            a=tuple(float(value) for value in payload["a"]),
            b=tuple(float(value) for value in payload["b"]),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class MeanAnchoredSoftplusDensityParameterProfile:
    source_evidence_id: str
    densities: tuple[float, ...]
    fitted_a: tuple[float, ...]
    b: tuple[float, ...]

    def __post_init__(self) -> None:
        densities = np.asarray(self.densities, dtype=np.float64)
        fitted_a = np.asarray(self.fitted_a, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)
        if (
            len(self.source_evidence_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_evidence_id
            )
            or densities.ndim != 1
            or densities.size < 2
            or fitted_a.shape != densities.shape
            or b.shape != densities.shape
            or not np.all(np.isfinite(densities))
            or not np.all(np.diff(densities) > 0.0)
            or not np.all(densities > 0.0)
            or not np.all(np.isfinite(fitted_a))
            or not np.all(np.isfinite(b))
            or not np.all(b > 0.0)
        ):
            raise ValueError("invalid mean-anchored softplus parameter profile")

    @staticmethod
    def _inverse_softplus(density: np.ndarray) -> np.ndarray:
        return np.log(np.expm1(density))

    def parameters(self, density: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(density, dtype=np.float64)
        nodes = np.asarray(self.densities, dtype=np.float64)
        if (
            not np.all(np.isfinite(requested))
            or np.any(requested < nodes[0])
            or np.any(requested > nodes[-1])
        ):
            raise ValueError("density is outside the mean-anchored profile")
        fitted_a = np.asarray(self.fitted_a, dtype=np.float64)
        source_b = np.asarray(self.b, dtype=np.float64)
        residual = fitted_a - self._inverse_softplus(nodes)
        result_a = self._inverse_softplus(requested) + np.interp(
            requested, nodes, residual
        )
        result_b = np.exp(np.interp(requested, nodes, np.log(source_b)))
        for node, value_a, value_b in zip(nodes, fitted_a, source_b, strict=True):
            result_a = np.where(requested == node, value_a, result_a)
            result_b = np.where(requested == node, value_b, result_b)
        return result_a, result_b

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "neuro-film.mean-anchored-softplus-density-parameter-profile.v1",
            "source_evidence_id": self.source_evidence_id,
            "densities": list(self.densities),
            "fitted_a": list(self.fitted_a),
            "b": list(self.b),
            "a_coordinate": (
                "inverse_softplus_density_plus_piecewise_linear_node_residual"
            ),
            "b_interpolation": "piecewise_log_linear_in_density",
            "density_extrapolation_allowed": False,
            "parameter_refit_allowed": False,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> MeanAnchoredSoftplusDensityParameterProfile:
        if (
            payload.get("schema")
            != "neuro-film.mean-anchored-softplus-density-parameter-profile.v1"
            or payload.get("a_coordinate")
            != "inverse_softplus_density_plus_piecewise_linear_node_residual"
            or payload.get("b_interpolation") != "piecewise_log_linear_in_density"
            or payload.get("density_extrapolation_allowed") is not False
            or payload.get("parameter_refit_allowed") is not False
        ):
            raise ValueError("unsupported mean-anchored parameter profile schema")
        return cls(
            source_evidence_id=str(payload["source_evidence_id"]),
            densities=tuple(float(value) for value in payload["densities"]),
            fitted_a=tuple(float(value) for value in payload["fitted_a"]),
            b=tuple(float(value) for value in payload["b"]),
        )

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


def softplus_density_field(unit_field: np.ndarray, *, a: float, b: float) -> np.ndarray:
    unit = np.asarray(unit_field, dtype=np.float64)
    if (
        unit.ndim != 2
        or not unit.size
        or not np.all(np.isfinite(unit))
        or not math.isfinite(a)
        or not math.isfinite(b)
        or b < 0.0
    ):
        raise ValueError("invalid positive density-field inputs")
    result = np.ascontiguousarray(np.logaddexp(0.0, a + b * unit), dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise RuntimeError("softplus density field is not finite and positive")
    result.setflags(write=False)
    return result


__all__ = [
    "PROFILE_SCHEMA",
    "MeanAnchoredSoftplusDensityParameterProfile",
    "SoftplusDensityParameterProfile",
    "softplus_density_field",
]
