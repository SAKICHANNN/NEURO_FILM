"""Explicit scan-linear black/white normalization into display-linear light."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np

from .contracts import PhysicalDomain, PhysicalDomainArray, PhysicalUnit
from .reversal_development import (
    GenericReversalDevelopment,
    develop_reversal_layer_exposure,
    reversal_development_contract,
    reversal_development_identity,
)
from .scanned_interpretation import (
    scan_interpretation_medium,
    scanner_profile_identity,
)
from .interpretation_medium import prepare_interpretation_medium
from .scanner import ScannerProfile


SCAN_SIGNAL_NORMALIZATION_SCHEMA = (
    "neuro_film.scan_signal_black_white_normalization.v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHANNELS = ("red", "green", "blue")


@dataclass(frozen=True)
class ScanSignalNormalization:
    black_scan_rgb: tuple[float, float, float]
    white_scan_rgb: tuple[float, float, float]
    reversal_operator_sha256: str
    scanner_profile_sha256: str
    scanner_stages: tuple[str, ...]
    endpoint_source: str = "synthetic_flat_fields"
    calibrated: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        black = tuple(float(value) for value in self.black_scan_rgb)
        white = tuple(float(value) for value in self.white_scan_rgb)
        if len(black) != 3 or len(white) != 3:
            raise ValueError("scan normalization endpoints require three channels")
        if not np.all(np.isfinite(black)) or not np.all(np.isfinite(white)):
            raise ValueError("scan normalization endpoints must be finite")
        if np.any(np.asarray(black) < 0.0) or np.any(np.asarray(white) > 1.0):
            raise ValueError("scan normalization endpoints must be in [0, 1]")
        if np.any(np.asarray(white) <= np.asarray(black)):
            raise ValueError("white scan endpoint must exceed black per channel")
        if (
            not _SHA256_RE.fullmatch(self.reversal_operator_sha256)
            or not _SHA256_RE.fullmatch(self.scanner_profile_sha256)
        ):
            raise ValueError("normalization identities must be SHA-256")
        stages = tuple(self.scanner_stages)
        if not stages or "noise" in stages:
            raise ValueError(
                "endpoint stages must be non-empty, deterministic and noise-free"
            )
        if self.endpoint_source != "synthetic_flat_fields":
            raise ValueError("v1 endpoints must come from synthetic flat fields")
        if self.calibrated is not False or self.production_eligible is not False:
            raise ValueError("generic scan normalization is research-only")
        object.__setattr__(self, "black_scan_rgb", black)
        object.__setattr__(self, "white_scan_rgb", white)
        object.__setattr__(self, "scanner_stages", stages)

    @property
    def separation_rgb(self) -> np.ndarray:
        values = np.asarray(self.white_scan_rgb) - np.asarray(
            self.black_scan_rgb
        )
        values.setflags(write=False)
        return values

    def apply(self, scan: PhysicalDomainArray) -> PhysicalDomainArray:
        state = scan.require(PhysicalDomain.SCAN_LINEAR)
        if state.channels != _CHANNELS:
            raise ValueError("scan normalization requires RGB channel order")
        values = state.values.astype(np.float64)
        black = np.asarray(self.black_scan_rgb)
        white = np.asarray(self.white_scan_rgb)
        if np.any(values < black) or np.any(values > white):
            raise ValueError(
                "scan signal is outside normalization endpoints; clipping forbidden"
            )
        normalized = (values - black) / (white - black)
        output = normalized.astype(state.values.dtype)
        return PhysicalDomainArray(
            output,
            PhysicalDomain.DISPLAY_LINEAR,
            PhysicalUnit.RELATIVE_DISPLAY_LIGHT,
            state.channels,
            state.scale,
        )

    def inverse(self, display: PhysicalDomainArray) -> PhysicalDomainArray:
        state = display.require(PhysicalDomain.DISPLAY_LINEAR)
        if state.channels != _CHANNELS:
            raise ValueError("scan normalization requires RGB channel order")
        values = state.values.astype(np.float64)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("display-linear normalization input must be in [0, 1]")
        black = np.asarray(self.black_scan_rgb)
        white = np.asarray(self.white_scan_rgb)
        scan = black + values * (white - black)
        return PhysicalDomainArray(
            scan.astype(state.values.dtype),
            PhysicalDomain.SCAN_LINEAR,
            PhysicalUnit.RELATIVE_SCAN_SIGNAL,
            state.channels,
            state.scale,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCAN_SIGNAL_NORMALIZATION_SCHEMA,
            "black_scan_rgb": list(self.black_scan_rgb),
            "white_scan_rgb": list(self.white_scan_rgb),
            "reversal_operator_sha256": self.reversal_operator_sha256,
            "scanner_profile_sha256": self.scanner_profile_sha256,
            "scanner_stages": list(self.scanner_stages),
            "endpoint_source": self.endpoint_source,
            "calibrated": self.calibrated,
            "production_eligible": self.production_eligible,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScanSignalNormalization":
        if payload.get("schema") != SCAN_SIGNAL_NORMALIZATION_SCHEMA:
            raise ValueError("unsupported scan normalization schema")
        return cls(
            tuple(payload["black_scan_rgb"]),
            tuple(payload["white_scan_rgb"]),
            str(payload["reversal_operator_sha256"]),
            str(payload["scanner_profile_sha256"]),
            tuple(payload["scanner_stages"]),
            str(payload["endpoint_source"]),
            payload["calibrated"],
            payload["production_eligible"],
        )


def scan_signal_normalization_identity(
    normalization: ScanSignalNormalization,
) -> str:
    if not isinstance(normalization, ScanSignalNormalization):
        raise TypeError("normalization must be ScanSignalNormalization")
    return hashlib.sha256(
        json.dumps(
            normalization.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def derive_scan_signal_normalization(
    reversal: GenericReversalDevelopment,
    scanner: ScannerProfile,
    *,
    flat_field_shape: tuple[int, int, int],
    scanner_stages: tuple[str, ...],
) -> ScanSignalNormalization:
    """Derive exact black/white responses from synthetic endpoint flat fields."""
    if not isinstance(reversal, GenericReversalDevelopment):
        raise TypeError("reversal must be GenericReversalDevelopment")
    if not isinstance(scanner, ScannerProfile):
        raise TypeError("scanner must be ScannerProfile")
    shape = tuple(int(value) for value in flat_field_shape)
    if len(shape) != 3 or shape[-1] != 3 or min(shape) <= 0:
        raise ValueError("flat_field_shape must be positive HxWx3")
    stages = tuple(scanner_stages)
    if not stages or "noise" in stages:
        raise ValueError("endpoint derivation must exclude scanner noise")
    endpoint_rows = []
    contract = reversal_development_contract(reversal)
    for exposure_value in (
        0.0,
        reversal.maximum_relative_layer_exposure,
    ):
        exposure = PhysicalDomainArray(
            np.full(shape, exposure_value, dtype=np.float64),
            PhysicalDomain.LAYER_EXPOSURE,
            PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
            _CHANNELS,
        )
        developed = develop_reversal_layer_exposure(
            exposure, reversal, contract
        )
        medium = prepare_interpretation_medium(developed)
        scan = scan_interpretation_medium(
            medium, scanner, pixel_pitch_um=1.0, stages=stages
        ).scan_linear.values
        center = scan[shape[0] // 2, shape[1] // 2]
        endpoint_rows.append(tuple(float(value) for value in center))
    return ScanSignalNormalization(
        endpoint_rows[0],
        endpoint_rows[1],
        reversal_development_identity(reversal),
        scanner_profile_identity(scanner),
        stages,
    )
