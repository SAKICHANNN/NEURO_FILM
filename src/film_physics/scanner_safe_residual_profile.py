"""Hash-versioned profiles for the optional scanner safe-residual runtime."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .scanner import (
    SCANNER_SAFE_RESIDUAL_RUNTIME_ID,
    ScannerSafeResidualReceipt,
    apply_scanner_safe_residual,
)

PROFILE_SCHEMA = "neuro-film.scanner-safe-residual-profile.v1"


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True)
class ScannerSafeResidualProfile:
    profile_id: str
    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    bias: tuple[float, float, float]
    evidence_sha256: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("scanner residual profile_id must be non-empty")
        if matrix.shape != (3, 3) or bias.shape != (3,) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
            raise ValueError("scanner residual profile coefficients must be finite 3x3+bias")
        if len(self.evidence_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.evidence_sha256):
            raise ValueError("scanner residual evidence hash must be lowercase SHA-256")

    def to_payload(self) -> dict[str, Any]:
        body = {
            "schema": PROFILE_SCHEMA,
            "runtime_id": SCANNER_SAFE_RESIDUAL_RUNTIME_ID,
            "profile_id": self.profile_id,
            "matrix": [list(row) for row in self.matrix],
            "bias": list(self.bias),
            "evidence_sha256": self.evidence_sha256,
        }
        return {**body, "profile_sha256": hashlib.sha256(_canonical(body)).hexdigest()}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ScannerSafeResidualProfile:
        expected = {"schema", "runtime_id", "profile_id", "matrix", "bias", "evidence_sha256", "profile_sha256"}
        if set(payload) != expected or payload.get("schema") != PROFILE_SCHEMA or payload.get("runtime_id") != SCANNER_SAFE_RESIDUAL_RUNTIME_ID:
            raise ValueError("scanner residual profile fields drift")
        body = {key: value for key, value in payload.items() if key != "profile_sha256"}
        if hashlib.sha256(_canonical(body)).hexdigest() != payload["profile_sha256"]:
            raise ValueError("scanner residual profile hash mismatch")
        profile = cls(
            profile_id=str(payload["profile_id"]),
            matrix=tuple(tuple(float(value) for value in row) for row in payload["matrix"]),
            bias=tuple(float(value) for value in payload["bias"]),
            evidence_sha256=str(payload["evidence_sha256"]),
        )
        if profile.to_payload() != payload:
            raise ValueError("scanner residual profile is not canonical")
        return profile


def apply_scanner_safe_residual_profile(
    source: np.ndarray,
    profile: ScannerSafeResidualProfile,
) -> tuple[np.ndarray, ScannerSafeResidualReceipt]:
    source_rgb = np.asarray(source)
    if source_rgb.dtype != np.float64:
        raise TypeError("scanner residual profile input must be float64")
    candidate = source_rgb @ np.asarray(profile.matrix, dtype=np.float64).T + np.asarray(profile.bias, dtype=np.float64)
    output, receipt = apply_scanner_safe_residual(source_rgb, candidate)
    if not math.isfinite(receipt.maximum_collinearity_error):
        raise RuntimeError("scanner residual profile produced invalid receipt")
    return output, receipt


def identity_scanner_safe_residual_profile(*, evidence_sha256: str) -> ScannerSafeResidualProfile:
    return ScannerSafeResidualProfile(
        profile_id="scanner-safe-residual-identity-v1",
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        bias=(0.0, 0.0, 0.0),
        evidence_sha256=evidence_sha256,
    )


__all__ = [
    "PROFILE_SCHEMA",
    "ScannerSafeResidualProfile",
    "apply_scanner_safe_residual_profile",
    "identity_scanner_safe_residual_profile",
]
