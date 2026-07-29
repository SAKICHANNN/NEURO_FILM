"""Research-only inverse characterization of scanner device RGB."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


MEASURED_SCANNER_CHARACTERIZATION_SCHEMA = (
    "neuro_film.measured_scanner_characterization_bundle.v1"
)
SOURCE_DOMAIN = "normalized-unknown-scanner-device-rgb"
OUTPUT_DOMAIN = "normalized-cie-xyz-d50"


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


@dataclass(frozen=True)
class MeasuredScannerCharacterization:
    scanner: str
    software: str
    target_set: int
    archive_sha256: str
    common_power: float
    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    fit_patch_count: int
    fit_support_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("scanner", "software"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if (
            isinstance(self.target_set, bool)
            or not isinstance(self.target_set, int)
            or self.target_set <= 0
        ):
            raise ValueError("target_set must be a positive integer")
        for field_name in ("archive_sha256", "fit_support_sha256"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field_name} must be lowercase SHA-256")
        if (
            not math.isfinite(self.common_power)
            or self.common_power < 0.5
            or self.common_power > 2.5
        ):
            raise ValueError("common_power must be finite in [0.5, 2.5]")
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
            or np.any(matrix < 0.0)
            or np.any(matrix > 4.0)
        ):
            raise ValueError("matrix must be finite nonnegative 3x3 in [0, 4]")
        if (
            isinstance(self.fit_patch_count, bool)
            or not isinstance(self.fit_patch_count, int)
            or self.fit_patch_count <= 0
        ):
            raise ValueError("fit_patch_count must be positive")

    @property
    def entry_id(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_payload())).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {
            "scanner": self.scanner,
            "software": self.software,
            "target_set": self.target_set,
            "archive_sha256": self.archive_sha256,
            "common_power": self.common_power,
            "matrix": [list(row) for row in self.matrix],
            "fit_patch_count": self.fit_patch_count,
            "fit_support_sha256": self.fit_support_sha256,
        }


@dataclass(frozen=True)
class MeasuredScannerCharacterizationBundle:
    contract_sha256: str
    evidence_report_sha256: str
    entries: tuple[MeasuredScannerCharacterization, ...]
    claim_ceiling: str
    source_domain: str = SOURCE_DOMAIN
    output_domain: str = OUTPUT_DOMAIN
    schema: str = MEASURED_SCANNER_CHARACTERIZATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MEASURED_SCANNER_CHARACTERIZATION_SCHEMA:
            raise ValueError("unsupported measured scanner bundle schema")
        if self.source_domain != SOURCE_DOMAIN or self.output_domain != OUTPUT_DOMAIN:
            raise ValueError("measured scanner bundle domain mismatch")
        for field_name in ("contract_sha256", "evidence_report_sha256"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field_name} must be lowercase SHA-256")
        if not isinstance(self.claim_ceiling, str) or not self.claim_ceiling:
            raise ValueError("claim_ceiling must be non-empty")
        if not self.entries:
            raise ValueError("measured scanner bundle requires entries")
        identities = [
            (
                entry.scanner,
                entry.software,
                entry.target_set,
                entry.archive_sha256,
            )
            for entry in self.entries
        ]
        if identities != sorted(identities) or len(set(identities)) != len(
            identities
        ):
            raise ValueError("bundle entries must have unique canonical order")

    @property
    def bundle_id(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_payload())).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contract_sha256": self.contract_sha256,
            "evidence_report_sha256": self.evidence_report_sha256,
            "source_domain": self.source_domain,
            "output_domain": self.output_domain,
            "entries": [entry.to_payload() for entry in self.entries],
            "claim_ceiling": self.claim_ceiling,
        }


def measured_scanner_bundle_from_payload(
    payload: dict[str, Any],
) -> MeasuredScannerCharacterizationBundle:
    allowed = {
        "schema",
        "contract_sha256",
        "evidence_report_sha256",
        "source_domain",
        "output_domain",
        "entries",
        "claim_ceiling",
    }
    if set(payload) != allowed or not isinstance(payload.get("entries"), list):
        raise ValueError("measured scanner bundle payload shape mismatch")
    entry_fields = {
        "scanner",
        "software",
        "target_set",
        "archive_sha256",
        "common_power",
        "matrix",
        "fit_patch_count",
        "fit_support_sha256",
    }
    entries = []
    for row in payload["entries"]:
        if not isinstance(row, dict) or set(row) != entry_fields:
            raise ValueError("measured scanner entry payload shape mismatch")
        entries.append(
            MeasuredScannerCharacterization(
                scanner=row["scanner"],
                software=row["software"],
                target_set=row["target_set"],
                archive_sha256=row["archive_sha256"],
                common_power=row["common_power"],
                matrix=tuple(tuple(value for value in line) for line in row["matrix"]),
                fit_patch_count=row["fit_patch_count"],
                fit_support_sha256=row["fit_support_sha256"],
            )
        )
    return MeasuredScannerCharacterizationBundle(
        schema=payload["schema"],
        contract_sha256=payload["contract_sha256"],
        evidence_report_sha256=payload["evidence_report_sha256"],
        source_domain=payload["source_domain"],
        output_domain=payload["output_domain"],
        entries=tuple(entries),
        claim_ceiling=payload["claim_ceiling"],
    )


def select_measured_scanner_characterization(
    bundle: MeasuredScannerCharacterizationBundle,
    *,
    scanner: str,
    software: str,
    target_set: int,
    archive_sha256: str,
) -> MeasuredScannerCharacterization:
    matches = [
        entry
        for entry in bundle.entries
        if entry.scanner == scanner
        and entry.software == software
        and entry.target_set == target_set
        and entry.archive_sha256 == archive_sha256
    ]
    if len(matches) != 1:
        raise ValueError("scanner characterization identity mismatch")
    return matches[0]


def apply_measured_scanner_characterization(
    values: np.ndarray,
    characterization: MeasuredScannerCharacterization,
) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("scanner device RGB must be float32 or float64")
    if source.ndim < 2 or source.shape[-1] != 3:
        raise ValueError("scanner device RGB must end in three channels")
    if (
        not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
    ):
        raise ValueError("scanner device RGB must be finite in [0, 1]")
    transformed = np.power(
        source.astype(np.float64, copy=False),
        characterization.common_power,
        dtype=np.float64,
    )
    output = transformed @ np.asarray(
        characterization.matrix, dtype=np.float64
    ).T
    if (
        not np.all(np.isfinite(output))
        or np.any(output < 0.0)
        or np.any(output > 1.5)
    ):
        raise ValueError("characterized D50 XYZ leaves the frozen domain")
    return output
