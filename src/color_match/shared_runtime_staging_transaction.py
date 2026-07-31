"""Atomic shared staging gated by factual four-target runtime evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from src.inference.render_contract import atomic_write_json, sha256_file

from .strict_json import strict_json_loads
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .files import (
    _cleanup_owned_files,
    _commit_staged_batch,
    _remember_owned_file,
    _stage_path,
)
from .shared_numeric_guard import (
    SharedNumericBatchGuardV1,
    validate_shared_numeric_batch_guard_v1,
)
from .shared_operator_batch import (
    PreparedSharedOperatorApplyV1,
    SharedOperatorBatchV1,
    validate_prepared_shared_operator_apply_v1,
    validate_shared_operator_batch_v1,
)
from .shared_product_authorization import (
    SharedProductStagingAuthorizationV1,
    validate_shared_product_staging_authorization_v1,
)
from .shared_runtime_qualification import (
    RuntimeQualifiedSharedAuthorizationV1,
    validate_runtime_qualified_shared_authorization_binding_v1,
)
from .shared_staging_transaction import (
    EXTERNAL_SHARED_STAGING_CLAIM_CEILING,
    EXTERNAL_SHARED_STAGING_SCHEMA_ID,
    ExternalSharedStagedOutputV1,
    ExternalSharedStagingRunV1,
    validate_external_shared_staging_run_v1,
)
from .staging_io import (
    _bounded_output_paths,
    encode_sdr_staging_output,
    validate_sdr_staging_destinations,
)
from .transaction_lock import (
    _HELD_LOCK_KEYS,
    _lock_key,
    target_transaction_lock as _target_transaction_lock,
)


RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID = (
    "neuro-film.runtime-qualified-shared-staging-run.v1"
)
RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING = (
    "runtime-qualified-shared-staging-files-committed-not-delivered"
)
_STATE = "committed-to-runtime-qualified-shared-staging"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_RUN_KEYS = {
    "schema_id",
    "run_id",
    "runtime_qualification_id",
    "runtime_evidence_id",
    "declaration_id",
    "authorization_id",
    "upstream_batch_id",
    "numeric_guard_batch_id",
    "operator_id",
    "reference_view_id",
    "source_count",
    "state",
    "outputs",
    "report_path",
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
    "output_format",
    "output_bit_depth",
    "encode_clipped_fraction",
}


@dataclass(frozen=True)
class RuntimeQualifiedExternalSharedStagingRunV1:
    """Canonical P62 report for runtime-qualified staged files."""

    schema_id: str
    run_id: str
    runtime_qualification_id: str
    runtime_evidence_id: str
    declaration_id: str
    authorization_id: str
    upstream_batch_id: str
    numeric_guard_batch_id: str
    operator_id: str
    reference_view_id: str
    source_count: int
    state: str
    outputs: tuple[ExternalSharedStagedOutputV1, ...]
    report_path: str
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class CommittedRuntimeQualifiedExternalSharedStagingV1:
    run: RuntimeQualifiedExternalSharedStagingRunV1
    report_file_sha256: str


def _strict(
    value: Any,
    keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from runtime-qualified shared staging"
        )
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _identity_payload(
    value: RuntimeQualifiedExternalSharedStagingRunV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("run_id")
    return payload


def _validate_runtime_qualified_input_bindings(
    *,
    batch: SharedOperatorBatchV1,
    numeric_guard: SharedNumericBatchGuardV1,
    authorization: SharedProductStagingAuthorizationV1,
    runtime_qualification: RuntimeQualifiedSharedAuthorizationV1,
    applies: Sequence[PreparedSharedOperatorApplyV1],
    expected_runtime_qualification_id: str,
) -> None:
    validate_shared_operator_batch_v1(batch)
    validate_shared_numeric_batch_guard_v1(numeric_guard)
    validate_shared_product_staging_authorization_v1(authorization)
    validate_runtime_qualified_shared_authorization_binding_v1(
        runtime_qualification,
        authorization=authorization,
    )
    _hash(
        expected_runtime_qualification_id,
        "expected_runtime_qualification_id",
    )
    if (
        runtime_qualification.state
        != "runtime-qualified-for-staging"
        or not runtime_qualification.runtime_ready
        or runtime_qualification.reasons
        or runtime_qualification.runtime_unsatisfied_targets
    ):
        raise ReferenceMatchContractError(
            "shared runtime staging requires exact runtime qualification"
        )
    if (
        runtime_qualification.qualification_id
        != expected_runtime_qualification_id
    ):
        raise ReferenceMatchContractError(
            "shared runtime staging qualification is not consumer-pinned"
        )
    if (
        numeric_guard.atomic_state != "eligible-for-transaction"
        or authorization.state != "authorized-for-staging"
        or numeric_guard.upstream_batch_id != batch.batch_id
        or numeric_guard.operator_id != batch.operator.operator_id
        or numeric_guard.source_count != batch.source_count
        or authorization.batch_id != batch.batch_id
        or authorization.numeric_guard_batch_id
        != numeric_guard.guard_batch_id
        or authorization.operator_id != batch.operator.operator_id
        or authorization.source_count != batch.source_count
    ):
        raise ReferenceMatchContractError(
            "shared runtime staging chain does not bind one batch"
        )
    if (
        not isinstance(applies, Sequence)
        or isinstance(applies, (str, bytes))
        or len(applies) != batch.source_count
    ):
        raise ReferenceMatchContractError(
            "shared runtime staging requires one apply per source"
        )
    for index, (receipt, decision, auth, prepared) in enumerate(
        zip(
            batch.sources,
            numeric_guard.decisions,
            authorization.sources,
            applies,
            strict=True,
        )
    ):
        validate_prepared_shared_operator_apply_v1(prepared)
        live = prepared.receipt
        if (
            receipt.source_index != index
            or decision.source_index != index
            or auth.source_index != index
            or live.source_index != index
            or live != receipt
            or decision.receipt_id != receipt.receipt_id
            or auth.receipt_id != receipt.receipt_id
            or decision.decision_id != auth.numeric_decision_id
            or not decision.accepted_for_transaction
            or not auth.numeric_accepted_for_transaction
            or auth.action != "authorized-for-staging"
            or live.operator_id != batch.operator.operator_id
            or live.source_view_id != auth.source_view_id
        ):
            raise ReferenceMatchContractError(
                f"shared runtime staging apply {index} "
                "does not bind the chain"
            )


def _snapshot_applies(
    applies: Sequence[PreparedSharedOperatorApplyV1],
) -> tuple[PreparedSharedOperatorApplyV1, ...]:
    snapshots: list[PreparedSharedOperatorApplyV1] = []
    for prepared in applies:
        pixels = np.array(
            prepared.pixels,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        pixels.flags.writeable = False
        snapshot = PreparedSharedOperatorApplyV1(
            receipt=prepared.receipt,
            pixels=pixels,
        )
        validate_prepared_shared_operator_apply_v1(snapshot)
        snapshots.append(snapshot)
    return tuple(snapshots)


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(
        attributes
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _reject_reparse_components(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for component in (absolute, *absolute.parents):
        if _is_reparse_point(component):
            raise ReferenceMatchContractError(
                f"{label} must not traverse a symlink or reparse point"
            )
    return absolute


def _runtime_output_paths(
    values: Iterable[Path | str],
    *,
    count: int,
) -> tuple[Path, ...]:
    paths = _bounded_output_paths(values, count=count)
    absolute = tuple(
        _reject_reparse_components(
            path,
            label="runtime-qualified output path",
        )
        for path in paths
    )
    keys = tuple(str(path).casefold() for path in absolute)
    if len(set(keys)) != len(keys):
        raise ReferenceMatchContractError("output paths must be unique")
    return absolute


def _require_absent_destination(
    path: Path,
    *,
    label: str,
) -> None:
    if os.path.lexists(path):
        raise ReferenceMatchContractError(
            f"{label} already exists; runtime staging is create-only"
        )


def commit_runtime_qualified_external_shared_staging_v1(
    *,
    batch: SharedOperatorBatchV1,
    numeric_guard: SharedNumericBatchGuardV1,
    authorization: SharedProductStagingAuthorizationV1,
    runtime_qualification: RuntimeQualifiedSharedAuthorizationV1,
    applies: Sequence[PreparedSharedOperatorApplyV1],
    output_paths: Iterable[Path | str],
    report_path: Path | str,
    expected_runtime_qualification_id: str,
    output_bit_depth: int = 16,
) -> CommittedRuntimeQualifiedExternalSharedStagingV1:
    """Create new outputs only after exact P61 qualification.

    Existing destination names are never replaced. The final publication uses
    an operating-system no-replace primitive, so a non-cooperating writer that
    wins after preflight is preserved. The report is published last and is the
    sole commit marker; an earlier failure may leave output orphans, but they
    are not a committed run and must not be consumed without P63 verification.
    """

    # This validation intentionally precedes path resolution, directory
    # creation, staging files and encoding.
    _validate_runtime_qualified_input_bindings(
        batch=batch,
        numeric_guard=numeric_guard,
        authorization=authorization,
        runtime_qualification=runtime_qualification,
        applies=applies,
        expected_runtime_qualification_id=(
            expected_runtime_qualification_id
        ),
    )
    snapshots = _snapshot_applies(applies)
    outputs = _runtime_output_paths(
        output_paths,
        count=batch.source_count,
    )
    report = _reject_reparse_components(
        Path(report_path),
        label="runtime-qualified report path",
    )
    validate_sdr_staging_destinations(
        outputs,
        report,
        output_bit_depth,
        label="runtime-qualified external shared staging",
    )
    with _target_transaction_lock((*outputs, report)):
        for index, output in enumerate(outputs):
            _require_absent_destination(
                output,
                label=f"runtime-qualified output {index}",
            )
        _require_absent_destination(
            report,
            label="runtime-qualified report",
        )

        token = uuid.uuid4().hex
        staged: list[Path] = []
        staged_identities: dict[Path, tuple[int, int]] = {}
        prepared_rows: list[ExternalSharedStagedOutputV1] = []
        try:
            for index, (prepared, output) in enumerate(
                zip(snapshots, outputs, strict=True)
            ):
                output.parent.mkdir(parents=True, exist_ok=True)
                stage = _stage_path(output, token)
                staged.append(stage)
                output_format, clipped_fraction = (
                    encode_sdr_staging_output(
                        prepared.pixels,
                        stage,
                        output_bit_depth=output_bit_depth,
                        label="runtime-qualified external shared",
                    )
                )
                _remember_owned_file(stage, staged_identities)
                receipt = prepared.receipt
                prepared_rows.append(
                    ExternalSharedStagedOutputV1(
                        source_index=index,
                        source_view_id=receipt.source_view_id,
                        apply_receipt_id=receipt.receipt_id,
                        producer_apply_result_id=(
                            receipt.producer_apply_result_id
                        ),
                        diagnostics_id=receipt.diagnostics_id,
                        output_view_id=receipt.output_view.view_id,
                        output_path=str(output),
                        output_file_sha256=sha256_file(stage),
                        output_format=output_format,
                        output_bit_depth=output_bit_depth,
                        encode_clipped_fraction=clipped_fraction,
                    )
                )
            provisional = RuntimeQualifiedExternalSharedStagingRunV1(
                schema_id=RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID,
                run_id="0" * 64,
                runtime_qualification_id=(
                    runtime_qualification.qualification_id
                ),
                runtime_evidence_id=(
                    runtime_qualification.runtime_evidence_id
                ),
                declaration_id=runtime_qualification.declaration_id,
                authorization_id=authorization.authorization_id,
                upstream_batch_id=batch.batch_id,
                numeric_guard_batch_id=numeric_guard.guard_batch_id,
                operator_id=batch.operator.operator_id,
                reference_view_id=batch.operator.reference_view_id,
                source_count=batch.source_count,
                state=_STATE,
                outputs=tuple(prepared_rows),
                report_path=str(report),
                claim_ceiling=(
                    RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING
                ),
            )
            run = replace(
                provisional,
                run_id=canonical_sha256(_identity_payload(provisional)),
            )
            validate_runtime_qualified_external_shared_staging_run_v1(
                run
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            staged_report = _stage_path(report, token)
            staged.append(staged_report)
            atomic_write_json(staged_report, run.to_dict())
            _remember_owned_file(staged_report, staged_identities)
            report_file_sha256 = sha256_file(staged_report)
            for stage, row in zip(
                (_stage_path(output, token) for output in outputs),
                prepared_rows,
                strict=True,
            ):
                if sha256_file(stage) != row.output_file_sha256:
                    raise ReferenceMatchContractError(
                        "runtime-qualified staged output changed before commit"
                    )
            if sha256_file(staged_report) != report_file_sha256:
                raise ReferenceMatchContractError(
                    "runtime-qualified staged report changed before commit"
                )
            pairs = tuple(
                (_stage_path(output, token), output)
                for output in outputs
            ) + ((staged_report, report),)
            for index, output in enumerate(outputs):
                _reject_reparse_components(
                    output,
                    label=f"runtime-qualified output path {index}",
                )
                _require_absent_destination(
                    output,
                    label=f"runtime-qualified output {index}",
                )
            _reject_reparse_components(
                report,
                label="runtime-qualified report path",
            )
            _require_absent_destination(
                report,
                label="runtime-qualified report",
            )
            expected_stage_hashes = {
                _stage_path(output, token): row.output_file_sha256
                for output, row in zip(
                    outputs,
                    prepared_rows,
                    strict=True,
                )
            }
            expected_stage_hashes[staged_report] = report_file_sha256
            expected_destination_hashes = {
                output: None for output in outputs
            }
            expected_destination_hashes[report] = None
            _commit_staged_batch(
                pairs,
                token=token,
                cleanup=staged,
                expected_stage_sha256=expected_stage_hashes,
                expected_destination_sha256=(
                    expected_destination_hashes
                ),
                replace_existing=False,
                targets_already_locked=True,
            )
            return CommittedRuntimeQualifiedExternalSharedStagingV1(
                run=run,
                report_file_sha256=report_file_sha256,
            )
        finally:
            _cleanup_owned_files(staged, staged_identities)


def _p50_validation_surrogate(
    value: RuntimeQualifiedExternalSharedStagingRunV1,
) -> ExternalSharedStagingRunV1:
    provisional = ExternalSharedStagingRunV1(
        schema_id=EXTERNAL_SHARED_STAGING_SCHEMA_ID,
        run_id="0" * 64,
        authorization_id=value.authorization_id,
        upstream_batch_id=value.upstream_batch_id,
        numeric_guard_batch_id=value.numeric_guard_batch_id,
        operator_id=value.operator_id,
        reference_view_id=value.reference_view_id,
        source_count=value.source_count,
        state="committed-to-shared-staging",
        outputs=value.outputs,
        report_path=value.report_path,
        claim_ceiling=EXTERNAL_SHARED_STAGING_CLAIM_CEILING,
    )
    identity = provisional.to_dict()
    identity.pop("run_id")
    return replace(
        provisional,
        run_id=canonical_sha256(identity),
    )


def validate_runtime_qualified_external_shared_staging_run_v1(
    value: RuntimeQualifiedExternalSharedStagingRunV1,
) -> None:
    if not isinstance(value, RuntimeQualifiedExternalSharedStagingRunV1):
        raise ReferenceMatchContractError(
            "runtime-qualified shared staging run type is invalid"
        )
    if value.schema_id != RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "runtime-qualified shared staging schema is invalid"
        )
    for field in (
        "run_id",
        "runtime_qualification_id",
        "runtime_evidence_id",
        "declaration_id",
        "authorization_id",
        "upstream_batch_id",
        "numeric_guard_batch_id",
        "operator_id",
        "reference_view_id",
    ):
        _hash(getattr(value, field), field)
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "runtime-qualified shared staging state is invalid"
        )
    if (
        value.claim_ceiling
        != RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "runtime-qualified shared staging claim ceiling mismatch"
        )
    path_texts = tuple(
        output.output_path for output in value.outputs
    ) + (value.report_path,)
    for path_text in path_texts:
        if not isinstance(path_text, str) or not path_text:
            raise ReferenceMatchContractError(
                "runtime-qualified shared staging path is invalid"
            )
        path = Path(path_text)
        if (
            not path.is_absolute()
            or str(Path(os.path.abspath(path_text))) != path_text
        ):
            raise ReferenceMatchContractError(
                "runtime-qualified shared staging paths "
                "must be canonical absolute paths"
            )
    # Delegate every P50 output/path/depth/cardinality invariant to the frozen
    # validator without assigning a P50 identity to this P62 report.
    validate_external_shared_staging_run_v1(
        _p50_validation_surrogate(value)
    )
    if value.run_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "runtime-qualified shared staging run identity mismatch"
        )


def runtime_qualified_external_shared_staging_run_to_json(
    value: RuntimeQualifiedExternalSharedStagingRunV1,
) -> str:
    validate_runtime_qualified_external_shared_staging_run_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def runtime_qualified_external_shared_staging_run_from_json(
    encoded: str,
) -> RuntimeQualifiedExternalSharedStagingRunV1:
    try:
        payload = strict_json_loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "runtime-qualified shared staging run is not valid JSON"
        ) from exc
    payload = _strict(
        payload,
        _RUN_KEYS,
        "runtime-qualified shared staging run",
    )
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "runtime-qualified shared staging outputs must be non-empty"
        )
    outputs: list[ExternalSharedStagedOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(
            raw,
            _OUTPUT_KEYS,
            f"runtime-qualified shared output {index}",
        )
        try:
            outputs.append(ExternalSharedStagedOutputV1(**dict(raw)))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "runtime-qualified shared staging output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        result = RuntimeQualifiedExternalSharedStagingRunV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "runtime-qualified shared staging run fields are invalid"
        ) from exc
    validate_runtime_qualified_external_shared_staging_run_v1(result)
    return result


__all__ = [
    "RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING",
    "RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID",
    "CommittedRuntimeQualifiedExternalSharedStagingV1",
    "RuntimeQualifiedExternalSharedStagingRunV1",
    "commit_runtime_qualified_external_shared_staging_v1",
    "runtime_qualified_external_shared_staging_run_from_json",
    "runtime_qualified_external_shared_staging_run_to_json",
    "validate_runtime_qualified_external_shared_staging_run_v1",
]
