"""Restart-safe verification of a committed external-core staging run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_staging_transaction import (
    ExternalCoreStagingRunV1,
    external_core_staging_run_from_json,
)
from .verification_io import read_hashed_utf8_report, verify_hashed_file


EXTERNAL_CORE_STAGING_VERIFICATION_SCHEMA_ID = (
    "neuro-film.external-core-staging-verification.v1"
)
EXTERNAL_CORE_STAGING_VERIFICATION_CLAIM_CEILING = (
    "verified-staging-not-delivered"
)
_STATE = "verified-staging"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "verification_id",
    "run_id",
    "report_path",
    "report_file_sha256",
    "authorization_id",
    "reference_intent_id",
    "source_count",
    "state",
    "outputs",
    "claim_ceiling",
}
_OUTPUT_KEYS = {
    "source_index",
    "apply_receipt_id",
    "output_path",
    "output_file_sha256",
}


@dataclass(frozen=True)
class VerifiedExternalCoreStagingOutputV1:
    source_index: int
    apply_receipt_id: str
    output_path: str
    output_file_sha256: str


@dataclass(frozen=True)
class ExternalCoreStagingVerificationV1:
    """Canonical proof that one P33 report and every staged file still match."""

    schema_id: str
    verification_id: str
    run_id: str
    report_path: str
    report_file_sha256: str
    authorization_id: str
    reference_intent_id: str
    source_count: int
    state: str
    outputs: tuple[VerifiedExternalCoreStagingOutputV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _strict(
    value: Any,
    expected: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferenceMatchContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ReferenceMatchContractError(
            f"{label} keys mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _identity_payload(
    value: ExternalCoreStagingVerificationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("verification_id")
    return payload


def _read_report(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[ExternalCoreStagingRunV1, str]:
    encoded, digest, _resolved = read_hashed_utf8_report(
        path,
        expected_sha256=expected_sha256,
        label="external staging report",
    )
    return external_core_staging_run_from_json(encoded), digest


def verify_external_core_staging_v1(
    *,
    report_path: Path | str,
    expected_report_sha256: str,
    expected_run_id: str,
) -> ExternalCoreStagingVerificationV1:
    """Reread and bind one complete P33 staging run without writing."""

    _sha256(expected_run_id, "expected_run_id")
    report = Path(report_path)
    run, report_digest = _read_report(
        report,
        expected_sha256=expected_report_sha256,
    )
    if run.run_id != expected_run_id:
        raise ReferenceMatchContractError(
            "external staging run identity mismatch"
        )
    resolved_report = str(report.resolve(strict=True))
    if (
        str(Path(run.report_path).resolve(strict=False)).casefold()
        != resolved_report.casefold()
    ):
        raise ReferenceMatchContractError(
            "external staging report path mismatch"
        )
    outputs: list[VerifiedExternalCoreStagingOutputV1] = []
    for row in run.outputs:
        path = Path(row.output_path)
        digest, resolved_output = verify_hashed_file(
            path,
            expected_sha256=row.output_file_sha256,
            label=f"external staged output {row.source_index}",
        )
        outputs.append(
            VerifiedExternalCoreStagingOutputV1(
                source_index=row.source_index,
                apply_receipt_id=row.apply_receipt_id,
                output_path=resolved_output,
                output_file_sha256=digest,
            )
        )
    provisional = ExternalCoreStagingVerificationV1(
        schema_id=EXTERNAL_CORE_STAGING_VERIFICATION_SCHEMA_ID,
        verification_id="0" * 64,
        run_id=run.run_id,
        report_path=resolved_report,
        report_file_sha256=report_digest,
        authorization_id=run.authorization_id,
        reference_intent_id=run.reference_intent_id,
        source_count=run.source_count,
        state=_STATE,
        outputs=tuple(outputs),
        claim_ceiling=(
            EXTERNAL_CORE_STAGING_VERIFICATION_CLAIM_CEILING
        ),
    )
    result = replace(
        provisional,
        verification_id=canonical_sha256(
            _identity_payload(provisional)
        ),
    )
    validate_external_core_staging_verification_v1(result)
    return result


def validate_external_core_staging_verification_v1(
    value: ExternalCoreStagingVerificationV1,
) -> None:
    if not isinstance(value, ExternalCoreStagingVerificationV1):
        raise ReferenceMatchContractError(
            "external staging verification type is invalid"
        )
    if value.schema_id != EXTERNAL_CORE_STAGING_VERIFICATION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported external staging verification schema"
        )
    _sha256(value.verification_id, "verification_id")
    _sha256(value.run_id, "run_id")
    _sha256(value.report_file_sha256, "report_file_sha256")
    _sha256(value.authorization_id, "authorization_id")
    _sha256(value.reference_intent_id, "reference_intent_id")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "external staging verification source_count mismatch"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "external staging verification state is unsupported"
        )
    if (
        value.claim_ceiling
        != EXTERNAL_CORE_STAGING_VERIFICATION_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "external staging verification claim ceiling mismatch"
        )
    if (
        not isinstance(value.report_path, str)
        or not value.report_path
        or Path(value.report_path).suffix.casefold() != ".json"
    ):
        raise ReferenceMatchContractError(
            "external staging verification report_path is invalid"
        )
    paths: list[str] = []
    for index, output in enumerate(value.outputs):
        if not isinstance(
            output,
            VerifiedExternalCoreStagingOutputV1,
        ):
            raise ReferenceMatchContractError(
                "external staging verification output type is invalid"
            )
        if output.source_index != index:
            raise ReferenceMatchContractError(
                "external staging verification indices must be contiguous"
            )
        _sha256(output.apply_receipt_id, "output.apply_receipt_id")
        _sha256(output.output_file_sha256, "output.output_file_sha256")
        if not isinstance(output.output_path, str) or not output.output_path:
            raise ReferenceMatchContractError(
                "external staging verification output_path is invalid"
            )
        paths.append(
            str(Path(output.output_path).resolve(strict=False)).casefold()
        )
    if len(set(paths)) != len(paths):
        raise ReferenceMatchContractError(
            "external staging verification output paths must be unique"
        )
    if str(
        Path(value.report_path).resolve(strict=False)
    ).casefold() in set(paths):
        raise ReferenceMatchContractError(
            "external staging verification report/output collision"
        )
    if value.verification_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "external staging verification_id mismatch"
        )


def external_core_staging_verification_to_json(
    value: ExternalCoreStagingVerificationV1,
) -> str:
    validate_external_core_staging_verification_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def external_core_staging_verification_from_json(
    encoded: str,
) -> ExternalCoreStagingVerificationV1:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "external staging verification is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "external staging verification")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "external staging verification outputs must be non-empty"
        )
    outputs: list[VerifiedExternalCoreStagingOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(
            raw,
            _OUTPUT_KEYS,
            f"external staging verification output {index}",
        )
        try:
            outputs.append(VerifiedExternalCoreStagingOutputV1(**raw))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "external staging verification output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        value = ExternalCoreStagingVerificationV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "external staging verification fields are invalid"
        ) from exc
    validate_external_core_staging_verification_v1(value)
    return value


__all__ = [
    "EXTERNAL_CORE_STAGING_VERIFICATION_CLAIM_CEILING",
    "EXTERNAL_CORE_STAGING_VERIFICATION_SCHEMA_ID",
    "ExternalCoreStagingVerificationV1",
    "VerifiedExternalCoreStagingOutputV1",
    "external_core_staging_verification_from_json",
    "external_core_staging_verification_to_json",
    "validate_external_core_staging_verification_v1",
    "verify_external_core_staging_v1",
]
