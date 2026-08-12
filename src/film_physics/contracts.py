"""Typed contracts for deterministic physical film image formation.

This module defines representation boundaries, not a calibrated film model.
Every conversion that changes physical meaning must name its input and output
domain; callers cannot silently relabel display RGB as exposure or density.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.preprocess.types import WorkingImage

PROFILE_BUNDLE_SCHEMA = "neuro_film.physical_film_profile_bundle.v1"
DOMAIN_ARRAY_SCHEMA = "neuro_film.physical_domain_array.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PhysicalDomain(str, Enum):
    SCENE_LINEAR = "scene-linear-relative-exposure"
    LAYER_EXPOSURE = "film-layer-linear-exposure"
    DEVELOPED_DENSITY = "developed-optical-density"
    TRANSMITTANCE = "film-transmittance"
    SCAN_LINEAR = "scanner-linear-signal"
    DISPLAY_LINEAR = "display-linear-light"
    DISPLAY_RGB = "display-encoded-rgb"


class PhysicalUnit(str, Enum):
    RELATIVE_SCENE_EXPOSURE = "relative-scene-exposure"
    RELATIVE_LAYER_EXPOSURE = "relative-layer-exposure"
    OPTICAL_DENSITY = "log10-optical-density"
    TRANSMITTANCE = "unitless-transmittance"
    RELATIVE_SCAN_SIGNAL = "relative-scan-linear-signal"
    RELATIVE_DISPLAY_LIGHT = "relative-display-linear-light"
    ENCODED_DISPLAY_CODE = "encoded-display-code"


DOMAIN_UNITS: dict[PhysicalDomain, PhysicalUnit] = {
    PhysicalDomain.SCENE_LINEAR: PhysicalUnit.RELATIVE_SCENE_EXPOSURE,
    PhysicalDomain.LAYER_EXPOSURE: PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
    PhysicalDomain.DEVELOPED_DENSITY: PhysicalUnit.OPTICAL_DENSITY,
    PhysicalDomain.TRANSMITTANCE: PhysicalUnit.TRANSMITTANCE,
    PhysicalDomain.SCAN_LINEAR: PhysicalUnit.RELATIVE_SCAN_SIGNAL,
    PhysicalDomain.DISPLAY_LINEAR: PhysicalUnit.RELATIVE_DISPLAY_LIGHT,
    PhysicalDomain.DISPLAY_RGB: PhysicalUnit.ENCODED_DISPLAY_CODE,
}

CANONICAL_DOMAIN_ORDER = (
    PhysicalDomain.SCENE_LINEAR,
    PhysicalDomain.LAYER_EXPOSURE,
    PhysicalDomain.DEVELOPED_DENSITY,
    PhysicalDomain.TRANSMITTANCE,
    PhysicalDomain.SCAN_LINEAR,
    PhysicalDomain.DISPLAY_LINEAR,
    PhysicalDomain.DISPLAY_RGB,
)


class QualityTier(str, Enum):
    PREVIEW = "preview"
    STANDARD = "standard"
    REFERENCE = "reference"


@dataclass(frozen=True)
class QualityTierSpec:
    tier: QualityTier
    arithmetic: str
    product_runtime_allowed: bool
    nominal_megapixels: tuple[int, ...]
    approximation_must_be_versioned: bool


QUALITY_TIERS: dict[QualityTier, QualityTierSpec] = {
    QualityTier.PREVIEW: QualityTierSpec(
        QualityTier.PREVIEW, "float32", True, (1, 2), True
    ),
    QualityTier.STANDARD: QualityTierSpec(
        QualityTier.STANDARD, "float32", True, (12, 24), True
    ),
    QualityTier.REFERENCE: QualityTierSpec(
        QualityTier.REFERENCE, "float64", False, (100,), False
    ),
}


@dataclass(frozen=True)
class PhysicalScale:
    """Sampling scale in physical film coordinates."""

    pixel_pitch_um: float

    def __post_init__(self) -> None:
        value = float(self.pixel_pitch_um)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("pixel_pitch_um must be finite and positive")
        object.__setattr__(self, "pixel_pitch_um", value)

    @property
    def samples_per_mm(self) -> float:
        return 1000.0 / self.pixel_pitch_um

    @property
    def nyquist_cycles_per_mm(self) -> float:
        return 500.0 / self.pixel_pitch_um


@dataclass(frozen=True)
class PhysicalDomainArray:
    """Immutable finite samples with one explicit physical meaning."""

    values: np.ndarray
    domain: PhysicalDomain
    unit: PhysicalUnit
    channels: tuple[str, str, str]
    scale: PhysicalScale | None = None
    _copy: InitVar[bool] = True

    def __post_init__(self, _copy: bool) -> None:
        try:
            domain = PhysicalDomain(self.domain)
            unit = PhysicalUnit(self.unit)
        except ValueError as exc:
            raise ValueError("unsupported physical domain or unit") from exc
        if DOMAIN_UNITS[domain] is not unit:
            raise ValueError(f"{domain.value} requires unit {DOMAIN_UNITS[domain].value}")
        array = np.asarray(self.values)
        if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("physical samples must be float32 or float64")
        if array.ndim < 2 or array.shape[-1] != 3 or array.size == 0:
            raise ValueError("physical samples must have non-empty shape (..., 3)")
        if not np.all(np.isfinite(array)):
            raise ValueError("physical samples must be finite")
        channels = tuple(self.channels)
        if len(channels) != 3 or any(
            not isinstance(item, str) or not item for item in channels
        ):
            raise ValueError("channels must contain three non-empty names")
        if domain in {
            PhysicalDomain.SCENE_LINEAR,
            PhysicalDomain.LAYER_EXPOSURE,
            PhysicalDomain.DEVELOPED_DENSITY,
            PhysicalDomain.SCAN_LINEAR,
        } and np.any(array < 0.0):
            raise ValueError(f"{domain.value} samples must be nonnegative")
        if domain is PhysicalDomain.TRANSMITTANCE and (
            np.any(array <= 0.0) or np.any(array > 1.0)
        ):
            raise ValueError("transmittance samples must be in (0, 1]")
        if not _copy and (not array.flags.c_contiguous or not array.flags.owndata):
            raise ValueError("adopted physical samples must own contiguous data")
        owned = np.array(array, copy=True, order="C") if _copy else array
        owned.setflags(write=False)
        object.__setattr__(self, "values", owned)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "channels", channels)

    @classmethod
    def adopt(
        cls,
        values: np.ndarray,
        domain: PhysicalDomain,
        unit: PhysicalUnit,
        channels: tuple[str, str, str],
        scale: PhysicalScale | None = None,
    ) -> PhysicalDomainArray:
        """Transfer one owned contiguous array into the immutable contract."""
        return cls(values, domain, unit, channels, scale, _copy=False)

    def require(self, domain: PhysicalDomain) -> PhysicalDomainArray:
        expected = PhysicalDomain(domain)
        if self.domain is not expected:
            raise ValueError(
                f"domain mismatch: required {expected.value}, received {self.domain.value}"
            )
        return self

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": DOMAIN_ARRAY_SCHEMA,
            "domain": self.domain.value,
            "unit": self.unit.value,
            "channels": list(self.channels),
            "dtype": self.values.dtype.name,
            "shape": list(self.values.shape),
            "pixel_pitch_um": None if self.scale is None else self.scale.pixel_pitch_um,
        }


def scene_exposure_from_working_image(working: WorkingImage) -> PhysicalDomainArray:
    """Admit only an explicitly scene-linear linear-sRGB WorkingImage."""

    if not isinstance(working, WorkingImage):
        raise TypeError("working must be a WorkingImage")
    if working.transfer_state != "scene_linear":
        raise ValueError("physical film ingress requires scene_linear WorkingImage")
    if working.working_space not in {"linear_srgb", "linear_srgb_d65"}:
        raise ValueError("v1 physical film ingress requires D65 linear sRGB")
    return PhysicalDomainArray(
        working.pixels,
        PhysicalDomain.SCENE_LINEAR,
        PhysicalUnit.RELATIVE_SCENE_EXPOSURE,
        ("red", "green", "blue"),
    )


def density_to_transmittance(
    density: PhysicalDomainArray,
) -> PhysicalDomainArray:
    density.require(PhysicalDomain.DEVELOPED_DENSITY)
    values = np.power(10.0, -density.values.astype(np.float64))
    dtype = density.values.dtype
    return PhysicalDomainArray(
        values.astype(dtype),
        PhysicalDomain.TRANSMITTANCE,
        PhysicalUnit.TRANSMITTANCE,
        density.channels,
        density.scale,
    )


def transmittance_to_density(
    transmittance: PhysicalDomainArray,
) -> PhysicalDomainArray:
    transmittance.require(PhysicalDomain.TRANSMITTANCE)
    values = -np.log10(transmittance.values.astype(np.float64))
    dtype = transmittance.values.dtype
    return PhysicalDomainArray(
        values.astype(dtype),
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
        transmittance.channels,
        transmittance.scale,
    )


@dataclass(frozen=True)
class ComponentBinding:
    component_id: str
    schema: str
    sha256: str
    input_domain: PhysicalDomain
    output_domain: PhysicalDomain

    def __post_init__(self) -> None:
        if (
            not isinstance(self.component_id, str)
            or not self.component_id
            or not isinstance(self.schema, str)
            or not self.schema
        ):
            raise ValueError("component identity and schema must be non-empty")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("component sha256 must be lowercase hexadecimal")
        input_domain = PhysicalDomain(self.input_domain)
        output_domain = PhysicalDomain(self.output_domain)
        if CANONICAL_DOMAIN_ORDER.index(output_domain) < CANONICAL_DOMAIN_ORDER.index(
            input_domain
        ):
            raise ValueError("component cannot move backwards through physical domains")
        object.__setattr__(self, "input_domain", input_domain)
        object.__setattr__(self, "output_domain", output_domain)

    def to_dict(self) -> dict[str, str]:
        return {
            "component_id": self.component_id,
            "schema": self.schema,
            "sha256": self.sha256,
            "input_domain": self.input_domain.value,
            "output_domain": self.output_domain.value,
        }


@dataclass(frozen=True)
class FilmProfileBundle:
    """Versioned component manifest with a strict claim ceiling."""

    profile_id: str
    claim_level: str
    stock_id: str
    process_id: str
    scanner_profile_id: str
    evidence_manifest_sha256: str | None
    components: tuple[ComponentBinding, ...]
    schema: str = PROFILE_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        components = tuple(self.components)
        object.__setattr__(self, "components", components)
        if self.schema != PROFILE_BUNDLE_SCHEMA:
            raise ValueError("unsupported physical film profile schema")
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if self.claim_level not in {
            "generic-physical-inspired",
            "look-approximation",
            "calibrated-reference",
        }:
            raise ValueError("unsupported profile claim_level")
        for value, label in (
            (self.stock_id, "stock_id"),
            (self.process_id, "process_id"),
            (self.scanner_profile_id, "scanner_profile_id"),
        ):
            if not value:
                raise ValueError(f"{label} must be explicit, using 'unknown' if absent")
        if not components:
            raise ValueError("profile must bind at least one component")
        if not all(isinstance(item, ComponentBinding) for item in components):
            raise TypeError("profile components must be ComponentBinding values")
        if len({item.component_id for item in components}) != len(components):
            raise ValueError("component_id values must be unique")
        if self.claim_level == "calibrated-reference":
            if "unknown" in {self.stock_id, self.process_id, self.scanner_profile_id}:
                raise ValueError("calibrated profile requires stock, process and scanner IDs")
            if not isinstance(
                self.evidence_manifest_sha256, str
            ) or not _SHA256_RE.fullmatch(
                self.evidence_manifest_sha256
            ):
                raise ValueError("calibrated profile requires an evidence manifest hash")
        elif self.evidence_manifest_sha256 is not None:
            if not isinstance(
                self.evidence_manifest_sha256, str
            ) or not _SHA256_RE.fullmatch(self.evidence_manifest_sha256):
                raise ValueError("evidence manifest sha256 must be lowercase hexadecimal")
        ordered = [
            CANONICAL_DOMAIN_ORDER.index(component.input_domain)
            for component in components
        ]
        if ordered != sorted(ordered):
            raise ValueError("profile components must follow canonical domain order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "claim_level": self.claim_level,
            "stock_id": self.stock_id,
            "process_id": self.process_id,
            "scanner_profile_id": self.scanner_profile_id,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "components": [item.to_dict() for item in self.components],
        }

    @property
    def bundle_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FilmProfileBundle:
        if set(payload) != {
            "schema",
            "profile_id",
            "claim_level",
            "stock_id",
            "process_id",
            "scanner_profile_id",
            "evidence_manifest_sha256",
            "components",
        }:
            raise ValueError("physical profile fields must match v1 exactly")
        components_list = []
        for item in payload["components"]:
            if not isinstance(item, dict) or set(item) != {
                "component_id",
                "schema",
                "sha256",
                "input_domain",
                "output_domain",
            }:
                raise ValueError("physical profile component fields must match v1 exactly")
            components_list.append(
                ComponentBinding(
                    str(item["component_id"]),
                    str(item["schema"]),
                    str(item["sha256"]),
                    PhysicalDomain(item["input_domain"]),
                    PhysicalDomain(item["output_domain"]),
                )
            )
        components = tuple(components_list)
        return cls(
            profile_id=str(payload["profile_id"]),
            claim_level=str(payload["claim_level"]),
            stock_id=str(payload["stock_id"]),
            process_id=str(payload["process_id"]),
            scanner_profile_id=str(payload["scanner_profile_id"]),
            evidence_manifest_sha256=payload["evidence_manifest_sha256"],
            components=components,
            schema=str(payload["schema"]),
        )


def coordinate_counter_u64(
    *,
    profile_sha256: str,
    seed: int,
    frame: int,
    x: int,
    y: int,
    layer: int,
) -> int:
    """Partition-independent counter value for stochastic physical fields."""

    if not isinstance(profile_sha256, str) or not _SHA256_RE.fullmatch(profile_sha256):
        raise ValueError("profile_sha256 must be lowercase hexadecimal")
    values: Iterable[int] = (seed, frame, x, y, layer)
    if any(not isinstance(value, int) for value in values):
        raise TypeError("counter coordinates must be integers")
    if seed < -(2**63) or seed >= 2**63:
        raise ValueError("seed must fit signed 64-bit")
    if any(value >= 2**64 for value in (frame, x, y, layer)):
        raise ValueError("counter coordinates must fit unsigned 64-bit")
    if frame < 0 or x < 0 or y < 0 or layer < 0:
        raise ValueError("frame, coordinates and layer must be nonnegative")
    payload = bytes.fromhex(profile_sha256) + struct.pack(
        "<qQQQQ", seed, frame, x, y, layer
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
