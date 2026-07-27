"""Atomic staging transaction for an authorized shared-operator batch."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from src.inference.render_contract import atomic_write_json, sha256_file

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .files import _commit_staged_batch, _stage_path
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
from .staging_io import (
    SDR16_OUTPUT_EXTENSIONS,
    SDR_OUTPUT_EXTENSIONS,
    encode_sdr_staging_output,
    staging_output_paths,
    validate_sdr_staging_destinations,
)


EXTERNAL_SHARED_STAGING_SCHEMA_ID = (
    "neuro-film.external-shared-staging-run.v1"
)
EXTERNAL_SHARED_STAGING_CLAIM_CEILING = (
    "shared-staging-files-committed-not-delivered"
)
_STATE = "committed-to-shared-staging"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PRODUCER_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUN_KEYS = {
    "schema_id",
    "run_id",
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
class ExternalSharedStagedOutputV1:
    source_index: int
    source_view_id: str
    apply_receipt_id: str
    producer_apply_result_id: str
    diagnostics_id: str
    output_view_id: str
    output_path: str
    output_file_sha256: str
    output_format: str
    output_bit_depth: int
    encode_clipped_fraction: float


@dataclass(frozen=True)
class ExternalSharedStagingRunV1:
    """Canonical report for files committed only to shared staging."""

    schema_id: str
    run_id: str
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
class CommittedExternalSharedStagingV1:
    run: ExternalSharedStagingRunV1
    report_file_sha256: str


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the shared staging contract"
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


def _identity_payload(value: ExternalSharedStagingRunV1) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("run_id")
    return payload


def _validate_input_bindings(
    *,
    batch: SharedOperatorBatchV1,
    numeric_guard: SharedNumericBatchGuardV1,
    authorization: SharedProductStagingAuthorizationV1,
    applies: Sequence[PreparedSharedOperatorApplyV1],
) -> None:
    validate_shared_operator_batch_v1(batch)
    validate_shared_numeric_batch_guard_v1(numeric_guard)
    validate_shared_product_staging_authorization_v1(authorization)
    if (
        numeric_guard.atomic_state != "eligible-for-transaction"
        or authorization.state != "authorized-for-staging"
    ):
        raise ReferenceMatchContractError(
            "shared staging requires an authorized complete batch"
        )
    if (
        numeric_guard.upstream_batch_id != batch.batch_id
        or numeric_guard.operator_id != batch.operator.operator_id
        or numeric_guard.source_count != batch.source_count
        or authorization.batch_id != batch.batch_id
        or authorization.numeric_guard_batch_id
        != numeric_guard.guard_batch_id
        or authorization.operator_id != batch.operator.operator_id
        or authorization.source_count != batch.source_count
    ):
        raise ReferenceMatchContractError(
            "shared staging chain does not bind one batch"
        )
    if (
        not isinstance(applies, Sequence)
        or isinstance(applies, (str, bytes))
        or len(applies) != batch.source_count
    ):
        raise ReferenceMatchContractError(
            "shared staging requires one apply per source"
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
                f"shared staging apply {index} does not bind the chain"
            )


def commit_external_shared_staging_v1(
    *,
    batch: SharedOperatorBatchV1,
    numeric_guard: SharedNumericBatchGuardV1,
    authorization: SharedProductStagingAuthorizationV1,
    applies: Sequence[PreparedSharedOperatorApplyV1],
    output_paths: Iterable[Path | str],
    report_path: Path | str,
    output_bit_depth: int = 16,
) -> CommittedExternalSharedStagingV1:
    """Commit exact authorized shared outputs and report as one transaction."""

    _validate_input_bindings(
        batch=batch,
        numeric_guard=numeric_guard,
        authorization=authorization,
        applies=applies,
    )
    outputs = staging_output_paths(
        output_paths, count=batch.source_count
    )
    report = Path(report_path)
    validate_sdr_staging_destinations(
        outputs,
        report,
        output_bit_depth,
        label="external shared staging",
    )

    token = uuid.uuid4().hex
    staged: list[Path] = []
    prepared_rows: list[ExternalSharedStagedOutputV1] = []
    try:
        for index, (prepared, output) in enumerate(
            zip(applies, outputs, strict=True)
        ):
            output.parent.mkdir(parents=True, exist_ok=True)
            stage = _stage_path(output, token)
            staged.append(stage)
            output_format, clipped_fraction = encode_sdr_staging_output(
                prepared.pixels,
                stage,
                output_bit_depth=output_bit_depth,
                label="external shared",
            )
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
                    output_path=str(output.resolve(strict=False)),
                    output_file_sha256=sha256_file(stage),
                    output_format=output_format,
                    output_bit_depth=output_bit_depth,
                    encode_clipped_fraction=clipped_fraction,
                )
            )
        provisional = ExternalSharedStagingRunV1(
            schema_id=EXTERNAL_SHARED_STAGING_SCHEMA_ID,
            run_id="0" * 64,
            authorization_id=authorization.authorization_id,
            upstream_batch_id=batch.batch_id,
            numeric_guard_batch_id=numeric_guard.guard_batch_id,
            operator_id=batch.operator.operator_id,
            reference_view_id=batch.operator.reference_view_id,
            source_count=batch.source_count,
            state=_STATE,
            outputs=tuple(prepared_rows),
            report_path=str(report.resolve(strict=False)),
            claim_ceiling=EXTERNAL_SHARED_STAGING_CLAIM_CEILING,
        )
        run = replace(
            provisional,
            run_id=canonical_sha256(_identity_payload(provisional)),
        )
        validate_external_shared_staging_run_v1(run)
        report.parent.mkdir(parents=True, exist_ok=True)
        staged_report = _stage_path(report, token)
        staged.append(staged_report)
        atomic_write_json(staged_report, run.to_dict())
        report_file_sha256 = sha256_file(staged_report)
        pairs = tuple(
            (_stage_path(output, token), output) for output in outputs
        ) + ((staged_report, report),)
        _commit_staged_batch(pairs, token=token, cleanup=staged)
        return CommittedExternalSharedStagingV1(
            run=run,
            report_file_sha256=report_file_sha256,
        )
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def validate_external_shared_staging_run_v1(
    value: ExternalSharedStagingRunV1,
) -> None:
    if not isinstance(value, ExternalSharedStagingRunV1):
        raise ReferenceMatchContractError(
            "external shared staging run type is invalid"
        )
    if value.schema_id != EXTERNAL_SHARED_STAGING_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "external shared staging schema is invalid"
        )
    for field in (
        "run_id",
        "authorization_id",
        "upstream_batch_id",
        "numeric_guard_batch_id",
        "operator_id",
        "reference_view_id",
    ):
        _hash(getattr(value, field), field)
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "external shared staging source_count is invalid"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "external shared staging state is invalid"
        )
    if (
        value.claim_ceiling
        != EXTERNAL_SHARED_STAGING_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "external shared staging claim ceiling mismatch"
        )
    if (
        not isinstance(value.report_path, str)
        or not value.report_path
        or Path(value.report_path).suffix.casefold() != ".json"
    ):
        raise ReferenceMatchContractError(
            "external shared staging report path is invalid"
        )
    output_paths: set[str] = set()
    source_ids: set[str] = set()
    receipt_ids: set[str] = set()
    result_ids: set[str] = set()
    output_depths: set[int] = set()
    for index, output in enumerate(value.outputs):
        if not isinstance(output, ExternalSharedStagedOutputV1):
            raise ReferenceMatchContractError(
                "external shared staging output type is invalid"
            )
        if output.source_index != index:
            raise ReferenceMatchContractError(
                "external shared staging source order is invalid"
            )
        for field in (
            "source_view_id",
            "apply_receipt_id",
            "output_view_id",
            "output_file_sha256",
        ):
            _hash(getattr(output, field), f"output.{field}")
        _producer_hash(
            output.producer_apply_result_id,
            "output.producer_apply_result_id",
        )
        _producer_hash(output.diagnostics_id, "output.diagnostics_id")
        if not isinstance(output.output_path, str) or not output.output_path:
            raise ReferenceMatchContractError(
                "external shared staging output path is invalid"
            )
        output_key = str(
            Path(output.output_path).resolve(strict=False)
        ).casefold()
        if (
            output_key in output_paths
            or output.source_view_id in source_ids
            or output.apply_receipt_id in receipt_ids
            or output.producer_apply_result_id in result_ids
        ):
            raise ReferenceMatchContractError(
                "external shared staging output identities must be distinct"
            )
        output_paths.add(output_key)
        source_ids.add(output.source_view_id)
        receipt_ids.add(output.apply_receipt_id)
        result_ids.add(output.producer_apply_result_id)
        if output.output_format not in {"PNG", "JPEG", "TIFF"}:
            raise ReferenceMatchContractError(
                "external shared staging output format is invalid"
            )
        if output.output_bit_depth not in {8, 16}:
            raise ReferenceMatchContractError(
                "external shared staging output depth is invalid"
            )
        output_depths.add(output.output_bit_depth)
        allowed = (
            SDR_OUTPUT_EXTENSIONS
            if output.output_bit_depth == 8
            else SDR16_OUTPUT_EXTENSIONS
        )
        if Path(output.output_path).suffix.casefold() not in allowed:
            raise ReferenceMatchContractError(
                "external shared staging extension/depth mismatch"
            )
        if (
            isinstance(output.encode_clipped_fraction, bool)
            or not isinstance(output.encode_clipped_fraction, (int, float))
            or not np.isfinite(float(output.encode_clipped_fraction))
            or not 0.0 <= float(output.encode_clipped_fraction) <= 1.0
        ):
            raise ReferenceMatchContractError(
                "external shared staging clipping fraction is invalid"
            )
    if len(output_depths) != 1:
        raise ReferenceMatchContractError(
            "external shared staging output depth must be uniform"
        )
    report_key = str(
        Path(value.report_path).resolve(strict=False)
    ).casefold()
    if report_key in output_paths:
        raise ReferenceMatchContractError(
            "external shared staging report must not overwrite an output"
        )
    if value.run_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "external shared staging run identity mismatch"
        )


def external_shared_staging_run_to_json(
    value: ExternalSharedStagingRunV1,
) -> str:
    validate_external_shared_staging_run_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def external_shared_staging_run_from_json(
    encoded: str,
) -> ExternalSharedStagingRunV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "external shared staging run is not valid JSON"
        ) from exc
    payload = _strict(payload, _RUN_KEYS, "shared staging run")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "external shared staging outputs must be non-empty"
        )
    outputs: list[ExternalSharedStagedOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"shared output {index}")
        try:
            outputs.append(ExternalSharedStagedOutputV1(**dict(raw)))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "external shared staging output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        result = ExternalSharedStagingRunV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "external shared staging run fields are invalid"
        ) from exc
    validate_external_shared_staging_run_v1(result)
    return result


__all__ = [
    "EXTERNAL_SHARED_STAGING_CLAIM_CEILING",
    "EXTERNAL_SHARED_STAGING_SCHEMA_ID",
    "CommittedExternalSharedStagingV1",
    "ExternalSharedStagedOutputV1",
    "ExternalSharedStagingRunV1",
    "commit_external_shared_staging_v1",
    "external_shared_staging_run_from_json",
    "external_shared_staging_run_to_json",
    "validate_external_shared_staging_run_v1",
]
