"""Process-local byte consumption captured from the exact P63 handle session."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .strict_json import strict_json_loads
from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .shared_runtime_staging_verification import (
    RuntimeQualifiedExternalSharedStagingVerificationV1,
    _verify_runtime_qualified_external_shared_staging_with_capture_v1,
    validate_runtime_qualified_external_shared_staging_verification_v1,
)


RUNTIME_QUALIFIED_SHARED_STAGING_CONSUMPTION_SCHEMA_ID = (
    "neuro-film.runtime-qualified-shared-staging-consumption.v1"
)
RUNTIME_QUALIFIED_SHARED_STAGING_CONSUMPTION_CLAIM_CEILING = (
    "process-local-immutable-byte-snapshot-only-"
    "no-path-reopen-persistence-or-delivery"
)
RUNTIME_QUALIFIED_SHARED_STAGING_SNAPSHOT_SCOPE = (
    "current-process-immutable-bytes-from-one-handle-session"
)
MAX_CAPTURED_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_CAPTURED_OUTPUT_AGGREGATE_BYTES = 512 * 1024 * 1024
_STATE = "captured-runtime-qualified-shared-staging-bytes"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_OUTPUT_KEYS = {
    "source_index",
    "source_view_id",
    "output_view_id",
    "output_file_sha256",
    "output_file_size_bytes",
    "handle_identity_id",
}
_KEYS = {
    "schema_id",
    "consumption_id",
    "verification_id",
    "run_id",
    "runtime_qualification_id",
    "handle_policy_id",
    "handle_identity_scheme",
    "source_count",
    "state",
    "snapshot_scope",
    "path_consumption_authorized",
    "persistent_snapshot_authorized",
    "delivery_authorized",
    "outputs",
    "claim_ceiling",
}


@dataclass(frozen=True)
class CapturedRuntimeQualifiedSharedStagingOutputV1:
    source_index: int
    source_view_id: str
    output_view_id: str
    output_file_sha256: str
    output_file_size_bytes: int
    handle_identity_id: str


@dataclass(frozen=True)
class RuntimeQualifiedSharedStagingConsumptionRecordV1:
    schema_id: str
    consumption_id: str
    verification_id: str
    run_id: str
    runtime_qualification_id: str
    handle_policy_id: str
    handle_identity_scheme: str
    source_count: int
    state: str
    snapshot_scope: str
    path_consumption_authorized: bool
    persistent_snapshot_authorized: bool
    delivery_authorized: bool
    outputs: tuple[CapturedRuntimeQualifiedSharedStagingOutputV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class RuntimeQualifiedSharedStagingByteSnapshotV1:
    """Non-serializable process-local bytes plus their persisted-safe record."""

    record: RuntimeQualifiedSharedStagingConsumptionRecordV1
    verification: RuntimeQualifiedExternalSharedStagingVerificationV1
    output_bytes: tuple[bytes, ...]


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the consumption contract"
        )
    return value


def _identity_payload(
    value: RuntimeQualifiedSharedStagingConsumptionRecordV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("consumption_id")
    return payload


def validate_runtime_qualified_shared_staging_consumption_record_v1(
    value: RuntimeQualifiedSharedStagingConsumptionRecordV1,
) -> None:
    if not isinstance(
        value, RuntimeQualifiedSharedStagingConsumptionRecordV1
    ):
        raise ReferenceMatchContractError(
            "runtime staging consumption record type is invalid"
        )
    if value.schema_id != (
        RUNTIME_QUALIFIED_SHARED_STAGING_CONSUMPTION_SCHEMA_ID
    ):
        raise ReferenceMatchContractError(
            "runtime staging consumption schema is invalid"
        )
    for field in (
        "consumption_id",
        "verification_id",
        "run_id",
        "runtime_qualification_id",
    ):
        _hash(getattr(value, field), field)
    if (
        value.state != _STATE
        or value.snapshot_scope
        != RUNTIME_QUALIFIED_SHARED_STAGING_SNAPSHOT_SCOPE
        or value.path_consumption_authorized is not False
        or value.persistent_snapshot_authorized is not False
        or value.delivery_authorized is not False
        or value.claim_ceiling
        != RUNTIME_QUALIFIED_SHARED_STAGING_CONSUMPTION_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "runtime staging consumption authority is invalid"
        )
    if (
        not isinstance(value.handle_policy_id, str)
        or not value.handle_policy_id
        or not isinstance(value.handle_identity_scheme, str)
        or not value.handle_identity_scheme
        or isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "runtime staging consumption metadata is invalid"
        )
    aggregate = 0
    source_ids: set[str] = set()
    view_ids: set[str] = set()
    identities: set[str] = set()
    for index, output in enumerate(value.outputs):
        if (
            not isinstance(
                output,
                CapturedRuntimeQualifiedSharedStagingOutputV1,
            )
            or output.source_index != index
            or isinstance(output.output_file_size_bytes, bool)
            or not isinstance(output.output_file_size_bytes, int)
            or output.output_file_size_bytes <= 0
            or output.output_file_size_bytes > MAX_CAPTURED_OUTPUT_BYTES
        ):
            raise ReferenceMatchContractError(
                "runtime staging captured output is invalid"
            )
        for field in (
            "source_view_id",
            "output_view_id",
            "output_file_sha256",
            "handle_identity_id",
        ):
            _hash(getattr(output, field), f"output.{field}")
        aggregate += output.output_file_size_bytes
        if aggregate > MAX_CAPTURED_OUTPUT_AGGREGATE_BYTES:
            raise ReferenceMatchContractError(
                "runtime staging captured outputs exceed aggregate budget"
            )
        if (
            output.source_view_id in source_ids
            or output.output_view_id in view_ids
            or output.handle_identity_id in identities
        ):
            raise ReferenceMatchContractError(
                "runtime staging captured output identities must be distinct"
            )
        source_ids.add(output.source_view_id)
        view_ids.add(output.output_view_id)
        identities.add(output.handle_identity_id)
    if value.consumption_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "runtime staging consumption identity mismatch"
        )


def validate_runtime_qualified_shared_staging_byte_snapshot_v1(
    value: RuntimeQualifiedSharedStagingByteSnapshotV1,
) -> None:
    if not isinstance(value, RuntimeQualifiedSharedStagingByteSnapshotV1):
        raise ReferenceMatchContractError(
            "runtime staging byte snapshot type is invalid"
        )
    validate_runtime_qualified_shared_staging_consumption_record_v1(
        value.record
    )
    validate_runtime_qualified_external_shared_staging_verification_v1(
        value.verification
    )
    if (
        value.record.verification_id != value.verification.verification_id
        or value.record.run_id != value.verification.run_id
        or value.record.runtime_qualification_id
        != value.verification.runtime_qualification_id
        or value.record.handle_policy_id
        != value.verification.handle_policy_id
        or value.record.handle_identity_scheme
        != value.verification.handle_identity_scheme
        or value.record.source_count != value.verification.source_count
        or len(value.output_bytes) != value.record.source_count
    ):
        raise ReferenceMatchContractError(
            "runtime staging byte snapshot binding mismatch"
        )
    for captured, verified, raw in zip(
        value.record.outputs,
        value.verification.outputs,
        value.output_bytes,
        strict=True,
    ):
        if not isinstance(raw, bytes):
            raise ReferenceMatchContractError(
                "runtime staging snapshot output must be immutable bytes"
            )
        if (
            captured.source_index != verified.source_index
            or captured.source_view_id != verified.source_view_id
            or captured.output_view_id != verified.output_view_id
            or captured.output_file_sha256
            != verified.output_file_sha256
            or captured.output_file_size_bytes
            != verified.output_file_size_bytes
            or captured.handle_identity_id != verified.handle_identity_id
            or len(raw) != captured.output_file_size_bytes
            or hashlib.sha256(raw).hexdigest()
            != captured.output_file_sha256
        ):
            raise ReferenceMatchContractError(
                "runtime staging snapshot bytes do not match verification"
            )


def capture_runtime_qualified_shared_staging_bytes_v1(
    *,
    report_path: Path | str,
    expected_report_sha256: str,
    expected_run_id: str,
    expected_runtime_qualification_id: str,
) -> RuntimeQualifiedSharedStagingByteSnapshotV1:
    verification, raw_outputs = (
        _verify_runtime_qualified_external_shared_staging_with_capture_v1(
            report_path=report_path,
            expected_report_sha256=expected_report_sha256,
            expected_run_id=expected_run_id,
            expected_runtime_qualification_id=(
                expected_runtime_qualification_id
            ),
            capture_output_bytes=True,
            maximum_capture_output_bytes=MAX_CAPTURED_OUTPUT_BYTES,
            maximum_capture_aggregate_bytes=(
                MAX_CAPTURED_OUTPUT_AGGREGATE_BYTES
            ),
        )
    )
    outputs = tuple(
        CapturedRuntimeQualifiedSharedStagingOutputV1(
            source_index=verified.source_index,
            source_view_id=verified.source_view_id,
            output_view_id=verified.output_view_id,
            output_file_sha256=verified.output_file_sha256,
            output_file_size_bytes=verified.output_file_size_bytes,
            handle_identity_id=verified.handle_identity_id,
        )
        for verified in verification.outputs
    )
    provisional = RuntimeQualifiedSharedStagingConsumptionRecordV1(
        schema_id=RUNTIME_QUALIFIED_SHARED_STAGING_CONSUMPTION_SCHEMA_ID,
        consumption_id="0" * 64,
        verification_id=verification.verification_id,
        run_id=verification.run_id,
        runtime_qualification_id=verification.runtime_qualification_id,
        handle_policy_id=verification.handle_policy_id,
        handle_identity_scheme=verification.handle_identity_scheme,
        source_count=verification.source_count,
        state=_STATE,
        snapshot_scope=RUNTIME_QUALIFIED_SHARED_STAGING_SNAPSHOT_SCOPE,
        path_consumption_authorized=False,
        persistent_snapshot_authorized=False,
        delivery_authorized=False,
        outputs=outputs,
        claim_ceiling=(
            RUNTIME_QUALIFIED_SHARED_STAGING_CONSUMPTION_CLAIM_CEILING
        ),
    )
    record = replace(
        provisional,
        consumption_id=canonical_sha256(_identity_payload(provisional)),
    )
    result = RuntimeQualifiedSharedStagingByteSnapshotV1(
        record=record,
        verification=verification,
        output_bytes=raw_outputs,
    )
    validate_runtime_qualified_shared_staging_byte_snapshot_v1(result)
    return result


def runtime_qualified_shared_staging_consumption_record_to_json(
    value: RuntimeQualifiedSharedStagingConsumptionRecordV1,
) -> str:
    validate_runtime_qualified_shared_staging_consumption_record_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def runtime_qualified_shared_staging_consumption_record_from_json(
    encoded: str,
) -> RuntimeQualifiedSharedStagingConsumptionRecordV1:
    try:
        payload = strict_json_loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "runtime staging consumption record is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "runtime staging consumption record")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "runtime staging consumption outputs must be non-empty"
        )
    outputs: list[CapturedRuntimeQualifiedSharedStagingOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"captured output {index}")
        try:
            outputs.append(
                CapturedRuntimeQualifiedSharedStagingOutputV1(**dict(raw))
            )
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "runtime staging captured output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        result = RuntimeQualifiedSharedStagingConsumptionRecordV1(
            **converted
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "runtime staging consumption record fields are invalid"
        ) from exc
    validate_runtime_qualified_shared_staging_consumption_record_v1(result)
    return result


__all__ = [
    "CapturedRuntimeQualifiedSharedStagingOutputV1",
    "MAX_CAPTURED_OUTPUT_AGGREGATE_BYTES",
    "MAX_CAPTURED_OUTPUT_BYTES",
    "RUNTIME_QUALIFIED_SHARED_STAGING_CONSUMPTION_CLAIM_CEILING",
    "RUNTIME_QUALIFIED_SHARED_STAGING_CONSUMPTION_SCHEMA_ID",
    "RUNTIME_QUALIFIED_SHARED_STAGING_SNAPSHOT_SCOPE",
    "RuntimeQualifiedSharedStagingByteSnapshotV1",
    "RuntimeQualifiedSharedStagingConsumptionRecordV1",
    "capture_runtime_qualified_shared_staging_bytes_v1",
    "runtime_qualified_shared_staging_consumption_record_from_json",
    "runtime_qualified_shared_staging_consumption_record_to_json",
    "validate_runtime_qualified_shared_staging_byte_snapshot_v1",
    "validate_runtime_qualified_shared_staging_consumption_record_v1",
]
