"""Restart-safe verification for a committed shared staging run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .shared_staging_transaction import (
    ExternalSharedStagingRunV1,
    external_shared_staging_run_from_json,
)
from .verification_io import read_hashed_utf8_report, verify_hashed_file


EXTERNAL_SHARED_STAGING_VERIFICATION_SCHEMA_ID = (
    "neuro-film.external-shared-staging-verification.v1"
)
EXTERNAL_SHARED_STAGING_VERIFICATION_CLAIM_CEILING = (
    "verified-shared-staging-not-delivered"
)
_STATE = "verified-shared-staging"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PRODUCER_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "verification_id",
    "run_id",
    "report_path",
    "report_file_sha256",
    "authorization_id",
    "numeric_guard_batch_id",
    "operator_id",
    "reference_view_id",
    "source_count",
    "state",
    "outputs",
    "claim_ceiling",
}
_OUTPUT_KEYS = {
    "source_index",
    "apply_receipt_id",
    "producer_apply_result_id",
    "output_path",
    "output_file_sha256",
}


@dataclass(frozen=True)
class VerifiedExternalSharedStagingOutputV1:
    source_index: int
    apply_receipt_id: str
    producer_apply_result_id: str
    output_path: str
    output_file_sha256: str


@dataclass(frozen=True)
class ExternalSharedStagingVerificationV1:
    """Canonical proof that one P50 report and all files still match."""

    schema_id: str
    verification_id: str
    run_id: str
    report_path: str
    report_file_sha256: str
    authorization_id: str
    numeric_guard_batch_id: str
    operator_id: str
    reference_view_id: str
    source_count: int
    state: str
    outputs: tuple[VerifiedExternalSharedStagingOutputV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from shared staging verification"
        )
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _producer_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _PRODUCER_HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be lowercase sha256:<hex>"
        )
    return value


def _identity_payload(
    value: ExternalSharedStagingVerificationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("verification_id")
    return payload


def _read_report(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[ExternalSharedStagingRunV1, str, str]:
    encoded, digest, resolved = read_hashed_utf8_report(
        path,
        expected_sha256=expected_sha256,
        label="external shared staging report",
    )
    return (
        external_shared_staging_run_from_json(encoded),
        digest,
        resolved,
    )


def verify_external_shared_staging_v1(
    *,
    report_path: Path | str,
    expected_report_sha256: str,
    expected_run_id: str,
    expected_authorization_id: str,
    expected_numeric_guard_batch_id: str,
    expected_operator_id: str,
) -> ExternalSharedStagingVerificationV1:
    """Reread and bind one complete P50 run without writing."""

    for value, label in (
        (expected_run_id, "expected_run_id"),
        (expected_authorization_id, "expected_authorization_id"),
        (
            expected_numeric_guard_batch_id,
            "expected_numeric_guard_batch_id",
        ),
        (expected_operator_id, "expected_operator_id"),
    ):
        _hash(value, label)
    report = Path(report_path)
    run, report_digest, resolved_report = _read_report(
        report,
        expected_sha256=expected_report_sha256,
    )
    if run.run_id != expected_run_id:
        raise ReferenceMatchContractError(
            "external shared staging run identity mismatch"
        )
    if run.authorization_id != expected_authorization_id:
        raise ReferenceMatchContractError(
            "external shared staging authorization identity mismatch"
        )
    if run.numeric_guard_batch_id != expected_numeric_guard_batch_id:
        raise ReferenceMatchContractError(
            "external shared staging numeric guard identity mismatch"
        )
    if run.operator_id != expected_operator_id:
        raise ReferenceMatchContractError(
            "external shared staging operator identity mismatch"
        )
    if (
        str(Path(run.report_path).resolve(strict=False)).casefold()
        != resolved_report.casefold()
    ):
        raise ReferenceMatchContractError(
            "external shared staging report path mismatch"
        )
    outputs: list[VerifiedExternalSharedStagingOutputV1] = []
    for row in run.outputs:
        digest, resolved_output = verify_hashed_file(
            Path(row.output_path),
            expected_sha256=row.output_file_sha256,
            label=f"external shared staged output {row.source_index}",
        )
        outputs.append(
            VerifiedExternalSharedStagingOutputV1(
                source_index=row.source_index,
                apply_receipt_id=row.apply_receipt_id,
                producer_apply_result_id=row.producer_apply_result_id,
                output_path=resolved_output,
                output_file_sha256=digest,
            )
        )
    provisional = ExternalSharedStagingVerificationV1(
        schema_id=EXTERNAL_SHARED_STAGING_VERIFICATION_SCHEMA_ID,
        verification_id="0" * 64,
        run_id=run.run_id,
        report_path=resolved_report,
        report_file_sha256=report_digest,
        authorization_id=run.authorization_id,
        numeric_guard_batch_id=run.numeric_guard_batch_id,
        operator_id=run.operator_id,
        reference_view_id=run.reference_view_id,
        source_count=run.source_count,
        state=_STATE,
        outputs=tuple(outputs),
        claim_ceiling=(
            EXTERNAL_SHARED_STAGING_VERIFICATION_CLAIM_CEILING
        ),
    )
    result = replace(
        provisional,
        verification_id=canonical_sha256(
            _identity_payload(provisional)
        ),
    )
    validate_external_shared_staging_verification_v1(result)
    return result


def validate_external_shared_staging_verification_v1(
    value: ExternalSharedStagingVerificationV1,
) -> None:
    if not isinstance(value, ExternalSharedStagingVerificationV1):
        raise ReferenceMatchContractError(
            "external shared staging verification type is invalid"
        )
    if (
        value.schema_id
        != EXTERNAL_SHARED_STAGING_VERIFICATION_SCHEMA_ID
    ):
        raise ReferenceMatchContractError(
            "external shared staging verification schema is invalid"
        )
    for field in (
        "verification_id",
        "run_id",
        "report_file_sha256",
        "authorization_id",
        "numeric_guard_batch_id",
        "operator_id",
        "reference_view_id",
    ):
        _hash(getattr(value, field), field)
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "external shared staging verification source_count is invalid"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "external shared staging verification state is invalid"
        )
    if (
        value.claim_ceiling
        != EXTERNAL_SHARED_STAGING_VERIFICATION_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "external shared staging verification claim ceiling mismatch"
        )
    if (
        not isinstance(value.report_path, str)
        or not value.report_path
        or Path(value.report_path).suffix.casefold() != ".json"
    ):
        raise ReferenceMatchContractError(
            "external shared staging verification report path is invalid"
        )
    paths: set[str] = set()
    receipts: set[str] = set()
    results: set[str] = set()
    for index, output in enumerate(value.outputs):
        if not isinstance(
            output, VerifiedExternalSharedStagingOutputV1
        ):
            raise ReferenceMatchContractError(
                "external shared staging verification output type is invalid"
            )
        if output.source_index != index:
            raise ReferenceMatchContractError(
                "external shared staging verification order is invalid"
            )
        _hash(output.apply_receipt_id, "output.apply_receipt_id")
        _producer_hash(
            output.producer_apply_result_id,
            "output.producer_apply_result_id",
        )
        _hash(output.output_file_sha256, "output.output_file_sha256")
        if not isinstance(output.output_path, str) or not output.output_path:
            raise ReferenceMatchContractError(
                "external shared staging verification output path is invalid"
            )
        path_key = str(
            Path(output.output_path).resolve(strict=False)
        ).casefold()
        if (
            path_key in paths
            or output.apply_receipt_id in receipts
            or output.producer_apply_result_id in results
        ):
            raise ReferenceMatchContractError(
                "external shared staging verification identities "
                "must be distinct"
            )
        paths.add(path_key)
        receipts.add(output.apply_receipt_id)
        results.add(output.producer_apply_result_id)
    if str(
        Path(value.report_path).resolve(strict=False)
    ).casefold() in paths:
        raise ReferenceMatchContractError(
            "external shared staging verification report/output collision"
        )
    if value.verification_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "external shared staging verification identity mismatch"
        )


def external_shared_staging_verification_to_json(
    value: ExternalSharedStagingVerificationV1,
) -> str:
    validate_external_shared_staging_verification_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def external_shared_staging_verification_from_json(
    encoded: str,
) -> ExternalSharedStagingVerificationV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "external shared staging verification is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "shared staging verification")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "external shared staging verification outputs must be non-empty"
        )
    outputs: list[VerifiedExternalSharedStagingOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"verified output {index}")
        try:
            outputs.append(
                VerifiedExternalSharedStagingOutputV1(**dict(raw))
            )
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "external shared staging verification output fields "
                "are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        result = ExternalSharedStagingVerificationV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "external shared staging verification fields are invalid"
        ) from exc
    validate_external_shared_staging_verification_v1(result)
    return result


__all__ = [
    "EXTERNAL_SHARED_STAGING_VERIFICATION_CLAIM_CEILING",
    "EXTERNAL_SHARED_STAGING_VERIFICATION_SCHEMA_ID",
    "ExternalSharedStagingVerificationV1",
    "VerifiedExternalSharedStagingOutputV1",
    "external_shared_staging_verification_from_json",
    "external_shared_staging_verification_to_json",
    "validate_external_shared_staging_verification_v1",
    "verify_external_shared_staging_v1",
]
