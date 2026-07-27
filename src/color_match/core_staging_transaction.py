"""Atomic consumer staging for already-authorized external-core outputs."""

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
from .core_apply_receipt import validate_prepared_core_apply_receipt
from .core_contracts import (
    MATCH_PROFILE_DISPLAY_SRGB,
    validate_capabilities,
    validate_diagnostics,
    validate_transform_bundle,
)
from .core_product_authorization import (
    CoreProductStagingAuthorizationV1,
    validate_core_product_staging_authorization_v1,
)
from .dpct_adapter import AdaptedDpctCandidateV2
from .dpct_batch import (
    DpctBatchResolutionV1,
    validate_dpct_batch_resolution_v1,
)
from .files import _commit_staged_batch, _stage_path
from .staging_io import (
    SDR16_OUTPUT_EXTENSIONS,
    SDR_OUTPUT_EXTENSIONS,
    encode_sdr_staging_output,
    staging_output_paths,
    validate_sdr_staging_destinations,
)


EXTERNAL_CORE_STAGING_SCHEMA_ID = (
    "neuro-film.external-core-staging-run.v1"
)
EXTERNAL_CORE_STAGING_CLAIM_CEILING = (
    "staging-files-committed-not-delivered"
)
_STATE = "committed-to-staging"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_RUN_KEYS = {
    "schema_id",
    "run_id",
    "authorization_id",
    "upstream_batch_id",
    "reference_intent_id",
    "source_count",
    "state",
    "outputs",
    "report_path",
    "claim_ceiling",
}
_OUTPUT_KEYS = {
    "source_index",
    "source_view_id",
    "transform_id",
    "apply_receipt_id",
    "output_view_id",
    "output_path",
    "output_file_sha256",
    "output_format",
    "output_bit_depth",
    "encode_clipped_fraction",
}


@dataclass(frozen=True)
class ExternalCoreStagedOutputV1:
    source_index: int
    source_view_id: str
    transform_id: str
    apply_receipt_id: str
    output_view_id: str
    output_path: str
    output_file_sha256: str
    output_format: str
    output_bit_depth: int
    encode_clipped_fraction: float


@dataclass(frozen=True)
class ExternalCoreStagingRunV1:
    """Canonical record for files committed only to product staging."""

    schema_id: str
    run_id: str
    authorization_id: str
    upstream_batch_id: str
    reference_intent_id: str
    source_count: int
    state: str
    outputs: tuple[ExternalCoreStagedOutputV1, ...]
    report_path: str
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class CommittedExternalCoreStagingV1:
    run: ExternalCoreStagingRunV1
    report_file_sha256: str


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


def _identity_payload(value: ExternalCoreStagingRunV1) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("run_id")
    return payload


def _validate_input_bindings(
    *,
    batch: DpctBatchResolutionV1,
    authorization: CoreProductStagingAuthorizationV1,
    candidates: Sequence[AdaptedDpctCandidateV2],
) -> str:
    validate_dpct_batch_resolution_v1(batch)
    validate_core_product_staging_authorization_v1(authorization)
    if (
        batch.atomic_state != "pending-product-guard"
        or authorization.state != "authorized-for-staging"
    ):
        raise ReferenceMatchContractError(
            "external staging requires an authorized complete batch"
        )
    if (
        authorization.upstream_batch_id != batch.batch_id
        or authorization.source_count != batch.source_count
    ):
        raise ReferenceMatchContractError(
            "external staging authorization does not bind batch"
        )
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or len(candidates) != batch.source_count
    ):
        raise ReferenceMatchContractError(
            "external staging requires one candidate per source"
        )
    reference_intent_id: str | None = None
    for index, (batch_row, auth_row, candidate) in enumerate(
        zip(batch.sources, authorization.sources, candidates, strict=True)
    ):
        if not isinstance(candidate, AdaptedDpctCandidateV2):
            raise ReferenceMatchContractError(
                f"external staging candidate {index} type is invalid"
            )
        validate_transform_bundle(candidate.transform)
        validate_capabilities(candidate.capabilities)
        validate_diagnostics(candidate.diagnostics)
        validate_prepared_core_apply_receipt(candidate.prepared_output)
        receipt = candidate.prepared_output.receipt
        if (
            batch_row.source_index != index
            or auth_row.source_index != index
            or auth_row.action != "authorized-for-staging"
            or auth_row.accepted_for_product_staging is not True
            or auth_row.source_view_id != batch_row.source_view_id
            or candidate.transform.source_view_id != batch_row.source_view_id
            or candidate.transform.reference_view_id
            != batch.reference_view_id
            or candidate.transform.transform_id
            != batch_row.consumer_transform_id
            or receipt.receipt_id != batch_row.apply_receipt_id
            or receipt.source_view_id != batch_row.source_view_id
            or receipt.reference_view_id != batch.reference_view_id
            or receipt.transform_id != candidate.transform.transform_id
            or receipt.output_view.profile_id
            != MATCH_PROFILE_DISPLAY_SRGB
        ):
            raise ReferenceMatchContractError(
                f"external staging candidate {index} does not bind batch"
            )
        if reference_intent_id is None:
            reference_intent_id = candidate.transform.intent_id
        elif candidate.transform.intent_id != reference_intent_id:
            raise ReferenceMatchContractError(
                "external staging candidates must share one reference intent"
            )
    if reference_intent_id is None:  # pragma: no cover - non-empty invariant.
        raise AssertionError("authorized batch must contain a candidate")
    return reference_intent_id


def commit_external_core_staging_v1(
    *,
    batch: DpctBatchResolutionV1,
    authorization: CoreProductStagingAuthorizationV1,
    candidates: Sequence[AdaptedDpctCandidateV2],
    output_paths: Iterable[Path | str],
    report_path: Path | str,
    output_bit_depth: int = 16,
) -> CommittedExternalCoreStagingV1:
    """Atomically commit authorized candidate files to staging, never delivery."""

    reference_intent_id = _validate_input_bindings(
        batch=batch,
        authorization=authorization,
        candidates=candidates,
    )
    outputs = staging_output_paths(
        output_paths, count=batch.source_count
    )
    report = Path(report_path)
    validate_sdr_staging_destinations(
        outputs,
        report,
        output_bit_depth,
        label="external staging",
    )

    token = uuid.uuid4().hex
    staged: list[Path] = []
    prepared: list[ExternalCoreStagedOutputV1] = []
    try:
        for index, (candidate, output) in enumerate(
            zip(candidates, outputs, strict=True)
        ):
            output.parent.mkdir(parents=True, exist_ok=True)
            stage = _stage_path(output, token)
            staged.append(stage)
            output_format, clipped_fraction = encode_sdr_staging_output(
                candidate.prepared_output.pixels,
                stage,
                output_bit_depth=output_bit_depth,
                label="external",
            )
            receipt = candidate.prepared_output.receipt
            prepared.append(
                ExternalCoreStagedOutputV1(
                    source_index=index,
                    source_view_id=receipt.source_view_id,
                    transform_id=receipt.transform_id,
                    apply_receipt_id=receipt.receipt_id,
                    output_view_id=receipt.output_view.view_id,
                    output_path=str(output.resolve(strict=False)),
                    output_file_sha256=sha256_file(stage),
                    output_format=output_format,
                    output_bit_depth=output_bit_depth,
                    encode_clipped_fraction=clipped_fraction,
                )
            )
        provisional = ExternalCoreStagingRunV1(
            schema_id=EXTERNAL_CORE_STAGING_SCHEMA_ID,
            run_id="0" * 64,
            authorization_id=authorization.authorization_id,
            upstream_batch_id=batch.batch_id,
            reference_intent_id=reference_intent_id,
            source_count=batch.source_count,
            state=_STATE,
            outputs=tuple(prepared),
            report_path=str(report.resolve(strict=False)),
            claim_ceiling=EXTERNAL_CORE_STAGING_CLAIM_CEILING,
        )
        run = replace(
            provisional,
            run_id=canonical_sha256(_identity_payload(provisional)),
        )
        validate_external_core_staging_run_v1(run)
        report.parent.mkdir(parents=True, exist_ok=True)
        staged_report = _stage_path(report, token)
        staged.append(staged_report)
        atomic_write_json(staged_report, run.to_dict())
        report_file_sha256 = sha256_file(staged_report)
        pairs = tuple(
            (_stage_path(output, token), output) for output in outputs
        ) + ((staged_report, report),)
        _commit_staged_batch(pairs, token=token, cleanup=staged)
        return CommittedExternalCoreStagingV1(
            run=run,
            report_file_sha256=report_file_sha256,
        )
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def validate_external_core_staging_run_v1(
    value: ExternalCoreStagingRunV1,
) -> None:
    if not isinstance(value, ExternalCoreStagingRunV1):
        raise ReferenceMatchContractError(
            "external staging run type is invalid"
        )
    if value.schema_id != EXTERNAL_CORE_STAGING_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported external staging run schema"
        )
    _sha256(value.run_id, "run_id")
    _sha256(value.authorization_id, "authorization_id")
    _sha256(value.upstream_batch_id, "upstream_batch_id")
    _sha256(value.reference_intent_id, "reference_intent_id")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "external staging source_count mismatch"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "external staging state is unsupported"
        )
    if value.claim_ceiling != EXTERNAL_CORE_STAGING_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "external staging claim ceiling mismatch"
        )
    if not isinstance(value.report_path, str) or not value.report_path:
        raise ReferenceMatchContractError(
            "external staging report_path must be non-empty"
        )
    if Path(value.report_path).suffix.casefold() != ".json":
        raise ReferenceMatchContractError(
            "external staging report_path must use .json"
        )
    output_paths: list[str] = []
    output_depths: list[int] = []
    for index, output in enumerate(value.outputs):
        if not isinstance(output, ExternalCoreStagedOutputV1):
            raise ReferenceMatchContractError(
                "external staging output row type is invalid"
            )
        if output.source_index != index:
            raise ReferenceMatchContractError(
                "external staging source indices must be contiguous"
            )
        _sha256(output.source_view_id, "output.source_view_id")
        _sha256(output.transform_id, "output.transform_id")
        _sha256(output.apply_receipt_id, "output.apply_receipt_id")
        _sha256(output.output_view_id, "output.output_view_id")
        _sha256(output.output_file_sha256, "output.output_file_sha256")
        if not isinstance(output.output_path, str) or not output.output_path:
            raise ReferenceMatchContractError(
                "external staging output_path must be non-empty"
            )
        output_key = str(
            Path(output.output_path).resolve(strict=False)
        ).casefold()
        output_paths.append(output_key)
        if output.output_format not in {"PNG", "JPEG", "TIFF"}:
            raise ReferenceMatchContractError(
                "external staging output format is unsupported"
            )
        if output.output_bit_depth not in {8, 16}:
            raise ReferenceMatchContractError(
                "external staging output bit depth is unsupported"
            )
        output_depths.append(output.output_bit_depth)
        suffix = Path(output.output_path).suffix.casefold()
        allowed = (
            SDR_OUTPUT_EXTENSIONS
            if output.output_bit_depth == 8
            else SDR16_OUTPUT_EXTENSIONS
        )
        if suffix not in allowed:
            raise ReferenceMatchContractError(
                "external staging output extension/depth mismatch"
            )
        if (
            not isinstance(output.encode_clipped_fraction, float)
            or not np.isfinite(output.encode_clipped_fraction)
            or not 0.0 <= output.encode_clipped_fraction <= 1.0
        ):
            raise ReferenceMatchContractError(
                "external staging encode clipping fraction is invalid"
            )
    if len(set(output_paths)) != len(output_paths):
        raise ReferenceMatchContractError(
            "external staging output paths must be unique"
        )
    if len(set(output_depths)) != 1:
        raise ReferenceMatchContractError(
            "external staging output bit depth must be uniform"
        )
    if str(Path(value.report_path).resolve(strict=False)).casefold() in set(
        output_paths
    ):
        raise ReferenceMatchContractError(
            "external staging report must not overwrite an output"
        )
    if value.run_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "external staging run_id mismatch"
        )


def external_core_staging_run_to_json(
    value: ExternalCoreStagingRunV1,
) -> str:
    validate_external_core_staging_run_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def external_core_staging_run_from_json(
    encoded: str,
) -> ExternalCoreStagingRunV1:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "external staging run is not valid JSON"
        ) from exc
    payload = _strict(payload, _RUN_KEYS, "external staging run")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "external staging outputs must be non-empty"
        )
    outputs: list[ExternalCoreStagedOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"external staging output {index}")
        try:
            outputs.append(ExternalCoreStagedOutputV1(**raw))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "external staging output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        value = ExternalCoreStagingRunV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "external staging run fields are invalid"
        ) from exc
    validate_external_core_staging_run_v1(value)
    return value


__all__ = [
    "EXTERNAL_CORE_STAGING_CLAIM_CEILING",
    "EXTERNAL_CORE_STAGING_SCHEMA_ID",
    "CommittedExternalCoreStagingV1",
    "ExternalCoreStagedOutputV1",
    "ExternalCoreStagingRunV1",
    "commit_external_core_staging_v1",
    "external_core_staging_run_from_json",
    "external_core_staging_run_to_json",
    "validate_external_core_staging_run_v1",
]
