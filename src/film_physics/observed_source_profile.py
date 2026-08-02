"""Typed non-renderable manufacturer source observations.

These profiles preserve sampled source curves and their evidence identities.
They deliberately contain no executable PSF, NPS, grain geometry or placement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

PROFILE_SCHEMA = "neuro_film.vision3_observed_stock_source_profile.v1"
BUNDLE_SCHEMA = "neuro_film.vision3_observed_stock_source_profile_bundle.v1"
CHANNELS = ("blue", "green", "red")
STOCKS = (
    "kodak_vision3_50d_5203_7203",
    "kodak_vision3_250d_5207_7207",
    "kodak_vision3_500t_5219_7219",
)
EXECUTION_AUTHORITY = "observed-source-prior-only-not-renderable"
MTF_DOMAIN = "spatial_frequency_cycles_per_mm_to_status_m_response_fraction"
GRANULARITY_DOMAIN = (
    "relative_log10_exposure_to_diffuse_rms_status_m_sigma_d_at_48_micrometres"
)


def _valid_identity(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ObservedCurve:
    """One finite sampled curve with no implied interpolation authority."""

    coordinates: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        coordinates = tuple(float(value) for value in self.coordinates)
        values = tuple(float(value) for value in self.values)
        if (
            len(coordinates) < 2
            or len(coordinates) != len(values)
            or not np.all(np.isfinite(coordinates))
            or not np.all(np.isfinite(values))
            or not np.all(np.diff(coordinates) > 0.0)
        ):
            raise ValueError("observed curve samples must be finite and strictly ordered")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, Any]:
        return {"coordinates": list(self.coordinates), "values": list(self.values)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObservedCurve:
        if set(payload) != {"coordinates", "values"}:
            raise ValueError("observed curve fields drifted")
        return cls(tuple(payload["coordinates"]), tuple(payload["values"]))


@dataclass(frozen=True)
class Vision3ObservedStockSourceProfile:
    """Observed MTF and diffuse-rms curves for one named VISION3 stock."""

    stock_id: str
    mtf_source_pdf_sha256: str
    mtf_source_graph_sha256: str
    granularity_source_pdf_sha256: str
    granularity_source_graph_sha256: str
    mtf_measurement_context: dict[str, Any]
    granularity_measurement_context: dict[str, Any]
    mtf_curves: dict[str, ObservedCurve]
    granularity_curves: dict[str, ObservedCurve]
    mtf_evidence_id: str
    granularity_evidence_id: str
    execution_authority: str = EXECUTION_AUTHORITY

    def __post_init__(self) -> None:
        if self.stock_id not in STOCKS:
            raise ValueError("unsupported observed stock")
        identities = (
            self.mtf_source_pdf_sha256,
            self.mtf_source_graph_sha256,
            self.granularity_source_pdf_sha256,
            self.granularity_source_graph_sha256,
            self.mtf_evidence_id,
            self.granularity_evidence_id,
        )
        if not all(_valid_identity(value) for value in identities):
            raise ValueError("observed source identities must be lowercase SHA-256")
        if self.execution_authority != EXECUTION_AUTHORITY:
            raise ValueError("observed source profile cannot authorize execution")
        mtf = dict(self.mtf_curves)
        granularity = dict(self.granularity_curves)
        if tuple(mtf) != CHANNELS or tuple(granularity) != CHANNELS:
            raise ValueError("observed source channel order drifted")
        if any(
            len(mtf[channel].coordinates) < 8
            or np.any(np.asarray(mtf[channel].values) < 0.0)
            or np.any(np.asarray(mtf[channel].values) > 1.1)
            for channel in CHANNELS
        ):
            raise ValueError("observed MTF curve is outside the source contract")
        if any(
            len(granularity[channel].coordinates) < 16
            or np.any(np.asarray(granularity[channel].values) < 0.001)
            or np.any(np.asarray(granularity[channel].values) > 0.05)
            for channel in CHANNELS
        ):
            raise ValueError("observed granularity curve is outside the source contract")
        mtf_context = dict(self.mtf_measurement_context)
        granularity_context = dict(self.granularity_measurement_context)
        if (
            mtf_context.get("spatial_frequency_unit") != "cycles/mm"
            or mtf_context.get("response_unit") != "percent"
            or granularity_context.get("aperture_micrometres") != 48.0
            or granularity_context.get("quantity") != "diffuse rms granularity Sigma D"
        ):
            raise ValueError("observed source measurement context drifted")
        object.__setattr__(self, "mtf_curves", MappingProxyType(mtf))
        object.__setattr__(self, "granularity_curves", MappingProxyType(granularity))
        object.__setattr__(self, "mtf_measurement_context", MappingProxyType(mtf_context))
        object.__setattr__(
            self, "granularity_measurement_context", MappingProxyType(granularity_context)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "stock_id": self.stock_id,
            "execution_authority": self.execution_authority,
            "curve_interpolation": "piecewise_linear_in_stored_coordinate_domain",
            "mtf_domain": MTF_DOMAIN,
            "granularity_domain": GRANULARITY_DOMAIN,
            "mtf_evidence_id": self.mtf_evidence_id,
            "granularity_evidence_id": self.granularity_evidence_id,
            "mtf_source": {
                "pdf_sha256": self.mtf_source_pdf_sha256,
                "graph_sha256": self.mtf_source_graph_sha256,
                "measurement_context": dict(self.mtf_measurement_context),
            },
            "granularity_source": {
                "pdf_sha256": self.granularity_source_pdf_sha256,
                "graph_sha256": self.granularity_source_graph_sha256,
                "measurement_context": dict(self.granularity_measurement_context),
            },
            "channel_order": list(CHANNELS),
            "mtf_curves": {
                channel: self.mtf_curves[channel].to_dict() for channel in CHANNELS
            },
            "granularity_curves": {
                channel: self.granularity_curves[channel].to_dict()
                for channel in CHANNELS
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Vision3ObservedStockSourceProfile:
        if (
            payload.get("schema") != PROFILE_SCHEMA
            or payload.get("channel_order") != list(CHANNELS)
            or payload.get("curve_interpolation")
            != "piecewise_linear_in_stored_coordinate_domain"
            or payload.get("mtf_domain") != MTF_DOMAIN
            or payload.get("granularity_domain") != GRANULARITY_DOMAIN
            or tuple(payload.get("mtf_curves", {})) != CHANNELS
            or tuple(payload.get("granularity_curves", {})) != CHANNELS
        ):
            raise ValueError("unsupported observed stock source profile schema")
        mtf_source = payload["mtf_source"]
        granularity_source = payload["granularity_source"]
        return cls(
            stock_id=str(payload["stock_id"]),
            mtf_source_pdf_sha256=str(mtf_source["pdf_sha256"]),
            mtf_source_graph_sha256=str(mtf_source["graph_sha256"]),
            granularity_source_pdf_sha256=str(granularity_source["pdf_sha256"]),
            granularity_source_graph_sha256=str(granularity_source["graph_sha256"]),
            mtf_measurement_context=dict(mtf_source["measurement_context"]),
            granularity_measurement_context=dict(
                granularity_source["measurement_context"]
            ),
            mtf_curves={
                channel: ObservedCurve.from_dict(payload["mtf_curves"][channel])
                for channel in CHANNELS
            },
            granularity_curves={
                channel: ObservedCurve.from_dict(
                    payload["granularity_curves"][channel]
                )
                for channel in CHANNELS
            },
            mtf_evidence_id=str(payload["mtf_evidence_id"]),
            granularity_evidence_id=str(payload["granularity_evidence_id"]),
            execution_authority=str(payload["execution_authority"]),
        )

    def identity(self) -> str:
        return _canonical_identity(self.to_dict())


@dataclass(frozen=True)
class Vision3ObservedSourceProfileBundle:
    """Three-stock observed-source bank with exact parent evidence bindings."""

    profiles: tuple[Vision3ObservedStockSourceProfile, ...]
    mtf_trace_sha256: str
    granularity_trace_sha256: str
    mtf_evidence_id: str
    granularity_evidence_id: str
    execution_authority: str = EXECUTION_AUTHORITY

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if tuple(profile.stock_id for profile in profiles) != STOCKS:
            raise ValueError("observed source profile stock order drifted")
        if not all(
            _valid_identity(value)
            for value in (
                self.mtf_trace_sha256,
                self.granularity_trace_sha256,
                self.mtf_evidence_id,
                self.granularity_evidence_id,
            )
        ):
            raise ValueError("observed source bundle identities are invalid")
        if self.execution_authority != EXECUTION_AUTHORITY:
            raise ValueError("observed source bundle cannot authorize execution")
        if any(
            profile.mtf_evidence_id != self.mtf_evidence_id
            or profile.granularity_evidence_id != self.granularity_evidence_id
            for profile in profiles
        ):
            raise ValueError("observed source profile evidence binding drifted")
        object.__setattr__(self, "profiles", profiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BUNDLE_SCHEMA,
            "execution_authority": self.execution_authority,
            "mtf_trace_sha256": self.mtf_trace_sha256,
            "granularity_trace_sha256": self.granularity_trace_sha256,
            "mtf_evidence_id": self.mtf_evidence_id,
            "granularity_evidence_id": self.granularity_evidence_id,
            "profiles": [profile.to_dict() for profile in self.profiles],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Vision3ObservedSourceProfileBundle:
        if payload.get("schema") != BUNDLE_SCHEMA:
            raise ValueError("unsupported observed source profile bundle schema")
        return cls(
            profiles=tuple(
                Vision3ObservedStockSourceProfile.from_dict(item)
                for item in payload["profiles"]
            ),
            mtf_trace_sha256=str(payload["mtf_trace_sha256"]),
            granularity_trace_sha256=str(payload["granularity_trace_sha256"]),
            mtf_evidence_id=str(payload["mtf_evidence_id"]),
            granularity_evidence_id=str(payload["granularity_evidence_id"]),
            execution_authority=str(payload["execution_authority"]),
        )

    def identity(self) -> str:
        return _canonical_identity(self.to_dict())


__all__ = [
    "BUNDLE_SCHEMA",
    "CHANNELS",
    "EXECUTION_AUTHORITY",
    "GRANULARITY_DOMAIN",
    "MTF_DOMAIN",
    "PROFILE_SCHEMA",
    "STOCKS",
    "ObservedCurve",
    "Vision3ObservedSourceProfileBundle",
    "Vision3ObservedStockSourceProfile",
]
