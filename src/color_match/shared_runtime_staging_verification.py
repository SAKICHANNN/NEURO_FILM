"""Handle-bound restart verification for one P62 staging commit."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .handle_verification_io import (
    HANDLE_POLICY_ID,
    MAX_STAGED_OUTPUT_AGGREGATE_BYTES,
    MAX_STAGED_OUTPUT_BYTES,
    POSIX_HANDLE_IDENTITY_SCHEME,
    WINDOWS_HANDLE_IDENTITY_SCHEME,
    StableFileHandleLease,
    open_stable_file_handle,
)
from .shared_runtime_staging_transaction import (
    RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING,
    RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID,
    RuntimeQualifiedExternalSharedStagingRunV1,
    runtime_qualified_external_shared_staging_run_from_json,
)
from .shared_staging_transaction import ExternalSharedStagedOutputV1
from .verification_io import MAX_REPORT_BYTES


RUNTIME_QUALIFIED_SHARED_STAGING_VERIFICATION_SCHEMA_ID = (
    "neuro-film.runtime-qualified-shared-staging-verification.v1"
)
RUNTIME_QUALIFIED_SHARED_STAGING_VERIFICATION_CLAIM_CEILING = (
    "runtime-qualified-shared-staging-handle-observation-only-"
    "no-path-consumption-or-delivery"
)
MAX_VERIFIED_STAGING_OUTPUTS = MAX_REFERENCE_MATCH_BATCH_SOURCES
_STATE = "verified-runtime-qualified-shared-staging"
_P62_STATE = "committed-to-runtime-qualified-shared-staging"
WINDOWS_HANDLE_OBSERVATION_SCOPE = (
    "windows-write-delete-share-denied-during-verification"
)
POSIX_HANDLE_OBSERVATION_SCOPE = (
    "posix-sequential-observation-no-write-exclusion"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PRODUCER_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_SCHEMES = {
    WINDOWS_HANDLE_IDENTITY_SCHEME,
    POSIX_HANDLE_IDENTITY_SCHEME,
}
_SCOPE_BY_SCHEME = {
    WINDOWS_HANDLE_IDENTITY_SCHEME: WINDOWS_HANDLE_OBSERVATION_SCOPE,
    POSIX_HANDLE_IDENTITY_SCHEME: POSIX_HANDLE_OBSERVATION_SCOPE,
}
_KEYS = {
    "schema_id",
    "verification_id",
    "handle_policy_id",
    "handle_identity_scheme",
    "handle_observation_scope",
    "path_consumption_authorized",
    "run_id",
    "runtime_qualification_id",
    "runtime_evidence_id",
    "declaration_id",
    "authorization_id",
    "upstream_batch_id",
    "numeric_guard_batch_id",
    "operator_id",
    "reference_view_id",
    "report_path",
    "report_file_sha256",
    "report_file_size_bytes",
    "report_handle_identity_id",
    "source_count",
    "state",
    "outputs",
    "claim_ceiling",
}
_OUTPUT_KEYS = {
    "source_index",
    "source_view_id",
    "apply_receipt_id",
    "producer_apply_result_id",
    "diagnostics_id",
    "output_view_id",
    "output_path",
    "output_file_sha256",
    "output_file_size_bytes",
    "handle_identity_id",
    "output_format",
    "output_bit_depth",
    "encode_clipped_fraction",
}


@dataclass(frozen=True)
class VerifiedRuntimeQualifiedExternalSharedStagingOutputV1:
    source_index: int
    source_view_id: str
    apply_receipt_id: str
    producer_apply_result_id: str
    diagnostics_id: str
    output_view_id: str
    output_path: str
    output_file_sha256: str
    output_file_size_bytes: int
    handle_identity_id: str
    output_format: str
    output_bit_depth: int
    encode_clipped_fraction: float


@dataclass(frozen=True)
class RuntimeQualifiedExternalSharedStagingVerificationV1:
    """Canonical P63 record from retained-handle sequential observations."""

    schema_id: str
    verification_id: str
    handle_policy_id: str
    handle_identity_scheme: str
    handle_observation_scope: str
    path_consumption_authorized: bool
    run_id: str
    runtime_qualification_id: str
    runtime_evidence_id: str
    declaration_id: str
    authorization_id: str
    upstream_batch_id: str
    numeric_guard_batch_id: str
    operator_id: str
    reference_view_id: str
    report_path: str
    report_file_sha256: str
    report_file_size_bytes: int
    report_handle_identity_id: str
    source_count: int
    state: str
    outputs: tuple[
        VerifiedRuntimeQualifiedExternalSharedStagingOutputV1, ...
    ]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from runtime-qualified "
            "shared staging verification"
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


def _bounded_positive_int(value: Any, maximum: int, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise ReferenceMatchContractError(f"{label} is outside its bound")
    return value


def _canonical_path_text(
    value: Any,
    *,
    label: str,
    suffix: str | None = None,
) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ReferenceMatchContractError(f"{label} is invalid")
    if os.path.abspath(value) != value:
        raise ReferenceMatchContractError(
            f"{label} must be a canonical absolute path"
        )
    if suffix is not None and Path(value).suffix.casefold() != suffix:
        raise ReferenceMatchContractError(f"{label} suffix is invalid")
    return value


def _path_key(value: str) -> str:
    return os.path.normcase(value)


def _identity_payload(
    value: RuntimeQualifiedExternalSharedStagingVerificationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("verification_id")
    return payload


def _p62_surrogate(
    value: RuntimeQualifiedExternalSharedStagingVerificationV1,
) -> RuntimeQualifiedExternalSharedStagingRunV1:
    outputs = tuple(
        ExternalSharedStagedOutputV1(
            source_index=output.source_index,
            source_view_id=output.source_view_id,
            apply_receipt_id=output.apply_receipt_id,
            producer_apply_result_id=output.producer_apply_result_id,
            diagnostics_id=output.diagnostics_id,
            output_view_id=output.output_view_id,
            output_path=output.output_path,
            output_file_sha256=output.output_file_sha256,
            output_format=output.output_format,
            output_bit_depth=output.output_bit_depth,
            encode_clipped_fraction=output.encode_clipped_fraction,
        )
        for output in value.outputs
    )
    return RuntimeQualifiedExternalSharedStagingRunV1(
        schema_id=RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID,
        run_id=value.run_id,
        runtime_qualification_id=value.runtime_qualification_id,
        runtime_evidence_id=value.runtime_evidence_id,
        declaration_id=value.declaration_id,
        authorization_id=value.authorization_id,
        upstream_batch_id=value.upstream_batch_id,
        numeric_guard_batch_id=value.numeric_guard_batch_id,
        operator_id=value.operator_id,
        reference_view_id=value.reference_view_id,
        source_count=value.source_count,
        state=_P62_STATE,
        outputs=outputs,
        report_path=value.report_path,
        claim_ceiling=RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING,
    )


def _validate_p62_identity_without_filesystem(
    value: RuntimeQualifiedExternalSharedStagingVerificationV1,
) -> None:
    surrogate = _p62_surrogate(value)
    payload = surrogate.to_dict()
    payload.pop("run_id")
    if surrogate.run_id != canonical_sha256(payload):
        raise ReferenceMatchContractError(
            "runtime-qualified shared staging run identity mismatch"
        )


def _decode_report(raw: bytes | None) -> RuntimeQualifiedExternalSharedStagingRunV1:
    if raw is None:
        raise ReferenceMatchContractError(
            "runtime-qualified staging report bytes are unavailable"
        )
    try:
        encoded = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ReferenceMatchContractError(
            "runtime-qualified staging report must be UTF-8"
        ) from exc
    return runtime_qualified_external_shared_staging_run_from_json(encoded)


def _verify_runtime_qualified_external_shared_staging_with_capture_v1(
    *,
    report_path: Path | str,
    expected_report_sha256: str,
    expected_run_id: str,
    expected_runtime_qualification_id: str,
    capture_output_bytes: bool,
    maximum_capture_output_bytes: int | None = None,
    maximum_capture_aggregate_bytes: int | None = None,
) -> tuple[
    RuntimeQualifiedExternalSharedStagingVerificationV1,
    tuple[bytes, ...],
]:
    """Verify one P62 run and optionally retain bytes from the same handles."""

    _hash(expected_report_sha256, "expected_report_sha256")
    _hash(expected_run_id, "expected_run_id")
    _hash(
        expected_runtime_qualification_id,
        "expected_runtime_qualification_id",
    )
    leases: list[StableFileHandleLease] = []
    with ExitStack() as stack:
        report_lease = stack.enter_context(
            open_stable_file_handle(
                report_path,
                label="runtime-qualified shared staging report",
            )
        )
        leases.append(report_lease)
        report_file = report_lease.verify(
            expected_sha256=expected_report_sha256,
            maximum_bytes=MAX_REPORT_BYTES,
            capture_bytes=True,
        )
        run = _decode_report(report_file.raw_bytes)
        if run.run_id != expected_run_id:
            raise ReferenceMatchContractError(
                "runtime-qualified shared staging run identity mismatch"
            )
        if (
            run.runtime_qualification_id
            != expected_runtime_qualification_id
        ):
            raise ReferenceMatchContractError(
                "runtime-qualified shared staging qualification "
                "identity mismatch"
            )
        if _path_key(run.report_path) != _path_key(report_file.path):
            raise ReferenceMatchContractError(
                "runtime-qualified shared staging report path mismatch"
            )
        _bounded_positive_int(
            run.source_count,
            MAX_VERIFIED_STAGING_OUTPUTS,
            "runtime-qualified shared staging source_count",
        )
        identities = {report_file.handle_identity_id}
        output_leases: list[
            tuple[ExternalSharedStagedOutputV1, StableFileHandleLease]
        ] = []
        aggregate_bytes = 0
        for row in run.outputs:
            lease = stack.enter_context(
                open_stable_file_handle(
                    row.output_path,
                    label=(
                        "runtime-qualified shared staged output "
                        f"{row.source_index}"
                    ),
                )
            )
            leases.append(lease)
            if (
                lease.identity_scheme
                != report_file.handle_identity_scheme
            ):
                raise ReferenceMatchContractError(
                    "runtime-qualified shared staging handle "
                    "identity schemes differ"
                )
            if lease.identity_id in identities:
                raise ReferenceMatchContractError(
                    "runtime-qualified shared staging handle "
                    "identities must be distinct"
                )
            identities.add(lease.identity_id)
            size = int(lease.initial_stat.st_size)
            if size <= 0 or size > MAX_STAGED_OUTPUT_BYTES:
                raise ReferenceMatchContractError(
                    "runtime-qualified shared staged output "
                    "violates its bounded size contract"
                )
            aggregate_bytes += size
            if aggregate_bytes > MAX_STAGED_OUTPUT_AGGREGATE_BYTES:
                raise ReferenceMatchContractError(
                    "runtime-qualified shared staged outputs "
                    "exceed the aggregate byte budget"
                )
            if capture_output_bytes and (
                maximum_capture_output_bytes is None
                or maximum_capture_aggregate_bytes is None
                or size > maximum_capture_output_bytes
                or aggregate_bytes > maximum_capture_aggregate_bytes
            ):
                raise ReferenceMatchContractError(
                    "runtime-qualified shared staged outputs "
                    "exceed the in-memory capture budget"
                )
            output_leases.append((row, lease))
        outputs: list[
            VerifiedRuntimeQualifiedExternalSharedStagingOutputV1
        ] = []
        captured_outputs: list[bytes] = []
        for row, lease in output_leases:
            verified = lease.verify(
                expected_sha256=row.output_file_sha256,
                maximum_bytes=(
                    maximum_capture_output_bytes
                    if capture_output_bytes
                    and maximum_capture_output_bytes is not None
                    else MAX_STAGED_OUTPUT_BYTES
                ),
                capture_bytes=capture_output_bytes,
            )
            if capture_output_bytes:
                if verified.raw_bytes is None:
                    raise ReferenceMatchContractError(
                        "runtime-qualified shared staged output "
                        "bytes are unavailable"
                    )
                captured_outputs.append(verified.raw_bytes)
            outputs.append(
                VerifiedRuntimeQualifiedExternalSharedStagingOutputV1(
                    source_index=row.source_index,
                    source_view_id=row.source_view_id,
                    apply_receipt_id=row.apply_receipt_id,
                    producer_apply_result_id=(
                        row.producer_apply_result_id
                    ),
                    diagnostics_id=row.diagnostics_id,
                    output_view_id=row.output_view_id,
                    output_path=row.output_path,
                    output_file_sha256=verified.sha256,
                    output_file_size_bytes=verified.size_bytes,
                    handle_identity_id=verified.handle_identity_id,
                    output_format=row.output_format,
                    output_bit_depth=row.output_bit_depth,
                    encode_clipped_fraction=row.encode_clipped_fraction,
                )
            )
        provisional = RuntimeQualifiedExternalSharedStagingVerificationV1(
            schema_id=(
                RUNTIME_QUALIFIED_SHARED_STAGING_VERIFICATION_SCHEMA_ID
            ),
            verification_id="0" * 64,
            handle_policy_id=HANDLE_POLICY_ID,
            handle_identity_scheme=report_file.handle_identity_scheme,
            handle_observation_scope=_SCOPE_BY_SCHEME[
                report_file.handle_identity_scheme
            ],
            path_consumption_authorized=False,
            run_id=run.run_id,
            runtime_qualification_id=run.runtime_qualification_id,
            runtime_evidence_id=run.runtime_evidence_id,
            declaration_id=run.declaration_id,
            authorization_id=run.authorization_id,
            upstream_batch_id=run.upstream_batch_id,
            numeric_guard_batch_id=run.numeric_guard_batch_id,
            operator_id=run.operator_id,
            reference_view_id=run.reference_view_id,
            report_path=run.report_path,
            report_file_sha256=report_file.sha256,
            report_file_size_bytes=report_file.size_bytes,
            report_handle_identity_id=report_file.handle_identity_id,
            source_count=run.source_count,
            state=_STATE,
            outputs=tuple(outputs),
            claim_ceiling=(
                RUNTIME_QUALIFIED_SHARED_STAGING_VERIFICATION_CLAIM_CEILING
            ),
        )
        result = replace(
            provisional,
            verification_id=canonical_sha256(
                _identity_payload(provisional)
            ),
        )
        validate_runtime_qualified_external_shared_staging_verification_v1(
            result
        )
        for lease in leases:
            lease.final_check()
        return result, tuple(captured_outputs)


def verify_runtime_qualified_external_shared_staging_v1(
    *,
    report_path: Path | str,
    expected_report_sha256: str,
    expected_run_id: str,
    expected_runtime_qualification_id: str,
) -> RuntimeQualifiedExternalSharedStagingVerificationV1:
    """Verify one P62 report and all outputs without writing or reopening data."""

    result, captured = (
        _verify_runtime_qualified_external_shared_staging_with_capture_v1(
            report_path=report_path,
            expected_report_sha256=expected_report_sha256,
            expected_run_id=expected_run_id,
            expected_runtime_qualification_id=(
                expected_runtime_qualification_id
            ),
            capture_output_bytes=False,
            maximum_capture_output_bytes=None,
            maximum_capture_aggregate_bytes=None,
        )
    )
    if captured:
        raise ReferenceMatchContractError(
            "verification-only operation retained unexpected output bytes"
        )
    return result


def validate_runtime_qualified_external_shared_staging_verification_v1(
    value: RuntimeQualifiedExternalSharedStagingVerificationV1,
) -> None:
    """Validate a P63 record without reading or resolving any filesystem path."""

    if not isinstance(
        value, RuntimeQualifiedExternalSharedStagingVerificationV1
    ):
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification type is invalid"
        )
    if (
        value.schema_id
        != RUNTIME_QUALIFIED_SHARED_STAGING_VERIFICATION_SCHEMA_ID
    ):
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification schema is invalid"
        )
    if value.handle_policy_id != HANDLE_POLICY_ID:
        raise ReferenceMatchContractError(
            "runtime-qualified staging handle policy is invalid"
        )
    if (
        not isinstance(value.handle_identity_scheme, str)
        or value.handle_identity_scheme not in _SCHEMES
    ):
        raise ReferenceMatchContractError(
            "runtime-qualified staging handle identity scheme is invalid"
        )
    if (
        value.handle_observation_scope
        != _SCOPE_BY_SCHEME[value.handle_identity_scheme]
    ):
        raise ReferenceMatchContractError(
            "runtime-qualified staging handle observation scope is invalid"
        )
    if value.path_consumption_authorized is not False:
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification cannot authorize "
            "path consumption"
        )
    for field in (
        "verification_id",
        "run_id",
        "runtime_qualification_id",
        "runtime_evidence_id",
        "declaration_id",
        "authorization_id",
        "upstream_batch_id",
        "numeric_guard_batch_id",
        "operator_id",
        "reference_view_id",
        "report_file_sha256",
        "report_handle_identity_id",
    ):
        _hash(getattr(value, field), field)
    if not isinstance(value.outputs, tuple):
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification outputs must be a tuple"
        )
    _bounded_positive_int(
        value.report_file_size_bytes,
        MAX_REPORT_BYTES,
        "report_file_size_bytes",
    )
    _bounded_positive_int(
        value.source_count,
        MAX_VERIFIED_STAGING_OUTPUTS,
        "source_count",
    )
    if value.source_count != len(value.outputs):
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification source_count mismatch"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification state is invalid"
        )
    if (
        value.claim_ceiling
        != RUNTIME_QUALIFIED_SHARED_STAGING_VERIFICATION_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification claim ceiling mismatch"
        )
    _canonical_path_text(
        value.report_path,
        label="runtime-qualified staging verification report path",
        suffix=".json",
    )
    paths = {_path_key(value.report_path)}
    identities = {value.report_handle_identity_id}
    source_ids: set[str] = set()
    receipt_ids: set[str] = set()
    result_ids: set[str] = set()
    depths: set[int] = set()
    aggregate_bytes = 0
    for index, output in enumerate(value.outputs):
        if not isinstance(
            output,
            VerifiedRuntimeQualifiedExternalSharedStagingOutputV1,
        ):
            raise ReferenceMatchContractError(
                "runtime-qualified staging verification output type is invalid"
            )
        if (
            isinstance(output.source_index, bool)
            or not isinstance(output.source_index, int)
            or output.source_index != index
        ):
            raise ReferenceMatchContractError(
                "runtime-qualified staging verification output order is invalid"
            )
        for field in (
            "source_view_id",
            "apply_receipt_id",
            "output_view_id",
            "output_file_sha256",
            "handle_identity_id",
        ):
            _hash(getattr(output, field), f"output.{field}")
        _producer_hash(
            output.producer_apply_result_id,
            "output.producer_apply_result_id",
        )
        _producer_hash(output.diagnostics_id, "output.diagnostics_id")
        output_size = _bounded_positive_int(
            output.output_file_size_bytes,
            MAX_STAGED_OUTPUT_BYTES,
            "output.output_file_size_bytes",
        )
        aggregate_bytes += output_size
        if aggregate_bytes > MAX_STAGED_OUTPUT_AGGREGATE_BYTES:
            raise ReferenceMatchContractError(
                "runtime-qualified staging verification outputs "
                "exceed the aggregate byte budget"
            )
        output_path = _canonical_path_text(
            output.output_path,
            label="runtime-qualified staging verification output path",
        )
        path_key = _path_key(output_path)
        if (
            path_key in paths
            or output.handle_identity_id in identities
            or output.source_view_id in source_ids
            or output.apply_receipt_id in receipt_ids
            or output.producer_apply_result_id in result_ids
        ):
            raise ReferenceMatchContractError(
                "runtime-qualified staging verification identities "
                "must be distinct"
            )
        paths.add(path_key)
        identities.add(output.handle_identity_id)
        source_ids.add(output.source_view_id)
        receipt_ids.add(output.apply_receipt_id)
        result_ids.add(output.producer_apply_result_id)
        if (
            not isinstance(output.output_format, str)
            or output.output_format not in {"PNG", "JPEG", "TIFF"}
        ):
            raise ReferenceMatchContractError(
                "runtime-qualified staging verification format is invalid"
            )
        if (
            isinstance(output.output_bit_depth, bool)
            or not isinstance(output.output_bit_depth, int)
            or output.output_bit_depth not in {8, 16}
        ):
            raise ReferenceMatchContractError(
                "runtime-qualified staging verification depth is invalid"
            )
        depths.add(output.output_bit_depth)
        suffix = Path(output_path).suffix.casefold()
        format_suffixes = {
            "PNG": {".png"},
            "JPEG": {".jpg", ".jpeg"},
            "TIFF": {".tif", ".tiff"},
        }
        if (
            suffix not in format_suffixes[output.output_format]
            or (
                output.output_format == "JPEG"
                and output.output_bit_depth != 8
            )
        ):
            raise ReferenceMatchContractError(
                "runtime-qualified staging verification "
                "extension/depth mismatch"
            )
        if (
            isinstance(output.encode_clipped_fraction, bool)
            or not isinstance(
                output.encode_clipped_fraction, (int, float)
            )
            or not math.isfinite(float(output.encode_clipped_fraction))
            or not 0.0
            <= float(output.encode_clipped_fraction)
            <= 1.0
        ):
            raise ReferenceMatchContractError(
                "runtime-qualified staging verification "
                "clipping fraction is invalid"
            )
    if len(depths) != 1:
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification depth must be uniform"
        )
    _validate_p62_identity_without_filesystem(value)
    if value.verification_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification identity mismatch"
        )


def runtime_qualified_external_shared_staging_verification_to_json(
    value: RuntimeQualifiedExternalSharedStagingVerificationV1,
) -> str:
    validate_runtime_qualified_external_shared_staging_verification_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def runtime_qualified_external_shared_staging_verification_from_json(
    encoded: str,
) -> RuntimeQualifiedExternalSharedStagingVerificationV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification is not valid JSON"
        ) from exc
    payload = _strict(
        payload,
        _KEYS,
        "runtime-qualified shared staging verification",
    )
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification outputs must be non-empty"
        )
    outputs: list[
        VerifiedRuntimeQualifiedExternalSharedStagingOutputV1
    ] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"verified runtime output {index}")
        try:
            outputs.append(
                VerifiedRuntimeQualifiedExternalSharedStagingOutputV1(
                    **dict(raw)
                )
            )
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "runtime-qualified staging verification output "
                "fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        result = RuntimeQualifiedExternalSharedStagingVerificationV1(
            **converted
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "runtime-qualified staging verification fields are invalid"
        ) from exc
    validate_runtime_qualified_external_shared_staging_verification_v1(result)
    return result


__all__ = [
    "MAX_VERIFIED_STAGING_OUTPUTS",
    "POSIX_HANDLE_OBSERVATION_SCOPE",
    "RUNTIME_QUALIFIED_SHARED_STAGING_VERIFICATION_CLAIM_CEILING",
    "RUNTIME_QUALIFIED_SHARED_STAGING_VERIFICATION_SCHEMA_ID",
    "RuntimeQualifiedExternalSharedStagingVerificationV1",
    "VerifiedRuntimeQualifiedExternalSharedStagingOutputV1",
    "WINDOWS_HANDLE_OBSERVATION_SCOPE",
    "runtime_qualified_external_shared_staging_verification_from_json",
    "runtime_qualified_external_shared_staging_verification_to_json",
    "validate_runtime_qualified_external_shared_staging_verification_v1",
    "verify_runtime_qualified_external_shared_staging_v1",
]
