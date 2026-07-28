"""Atomic one-reference/N-source intake for verified D-PCT outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any, Mapping, Sequence

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_acceptance import (
    CoreAcceptanceDecisionV1,
    validate_core_acceptance_decision,
)
from .core_adapter import (
    PreparedMatchViewV1,
    validate_prepared_match_view,
)
from .core_apply_receipt import (
    validate_core_apply_receipt_binding,
    validate_prepared_core_apply_receipt,
)
from .core_candidate_admission import (
    CoreCandidateAdmissionV2,
    validate_core_candidate_admission_binding,
)
from .dpct_adapter import (
    DPCT_COMPATIBILITY_PROFILE_ID,
    DPCT_PINNED_COMMIT,
    AdaptedDpctCandidateV2,
    DpctProducerFailureV2,
    dpct_producer_view_id_for_prepared_v2,
    validate_dpct_producer_failure_v2,
)


DPCT_BATCH_RESOLUTION_SCHEMA_ID = "neuro-film.dpct-batch-resolution.v1"
DPCT_BATCH_POLICY_ID = "neuro-film.dpct-atomic-batch.v1"
_REQUIRED_GATES = ("A1", "A4", "A5")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PRODUCER_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_RESOLUTION_KEYS = {
    "schema_id",
    "batch_id",
    "policy_id",
    "compatibility_profile_id",
    "producer_commit",
    "reference_view_id",
    "producer_reference_view_id",
    "source_count",
    "atomic_state",
    "required_gates",
    "sources",
}
_SOURCE_KEYS = {
    "source_index",
    "source_view_id",
    "producer_source_view_id",
    "producer_status",
    "producer_outcome_id",
    "consumer_transform_id",
    "apply_receipt_id",
    "acceptance_decision_id",
    "admission_id",
    "individual_action",
}


@dataclass(frozen=True)
class DpctBatchSourceResolutionV1:
    source_index: int
    source_view_id: str
    producer_source_view_id: str
    producer_status: str
    producer_outcome_id: str
    consumer_transform_id: str | None
    apply_receipt_id: str | None
    acceptance_decision_id: str | None
    admission_id: str | None
    individual_action: str


@dataclass(frozen=True)
class DpctBatchResolutionV1:
    """Atomic batch decision; never a delivered or applied result."""

    schema_id: str
    batch_id: str
    policy_id: str
    compatibility_profile_id: str
    producer_commit: str
    reference_view_id: str
    producer_reference_view_id: str
    source_count: int
    atomic_state: str
    required_gates: tuple[str, ...]
    sources: tuple[DpctBatchSourceResolutionV1, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_gates"] = list(self.required_gates)
        payload["sources"] = [asdict(source) for source in self.sources]
        return payload


DpctBatchOutcomeV1 = AdaptedDpctCandidateV2 | DpctProducerFailureV2
DpctBatchAdjudicationV1 = tuple[
    CoreAcceptanceDecisionV1,
    CoreCandidateAdmissionV2,
]


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


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _producer_hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _PRODUCER_HASH.fullmatch(value) is None
    ):
        raise ReferenceMatchContractError(
            f"{label} must be lowercase sha256:<hex>"
        )
    return value


def _optional_hash(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _hash(value, label)


def _identity_payload(value: DpctBatchResolutionV1) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("batch_id")
    return payload


def _validate_candidate_binding(
    candidate: AdaptedDpctCandidateV2,
    *,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    producer_source_view_id: str,
    producer_reference_view_id: str,
) -> None:
    if not isinstance(candidate, AdaptedDpctCandidateV2):
        raise ReferenceMatchContractError(
            "D-PCT candidate outcome type is invalid"
        )
    aliases = candidate.aliases
    if (
        aliases.compatibility_profile_id != DPCT_COMPATIBILITY_PROFILE_ID
        or aliases.producer_commit != DPCT_PINNED_COMMIT
        or aliases.source_view_id != producer_source_view_id
        or aliases.reference_view_id != producer_reference_view_id
    ):
        raise ReferenceMatchContractError(
            "D-PCT candidate producer identity mismatch"
        )
    if (
        candidate.transform.source_view_id != source.descriptor.view_id
        or candidate.transform.reference_view_id
        != reference.descriptor.view_id
    ):
        raise ReferenceMatchContractError(
            "D-PCT candidate consumer view binding mismatch"
        )
    validate_prepared_core_apply_receipt(candidate.prepared_output)
    validate_core_apply_receipt_binding(
        candidate.prepared_output.receipt,
        source=source.descriptor,
        reference=reference.descriptor,
        transform=candidate.transform,
        capabilities=candidate.capabilities,
        diagnostics=candidate.diagnostics,
    )
    if (
        candidate.prepared_output.receipt.receipt_id is None
        or candidate.aliases.result_id is None
    ):
        raise ReferenceMatchContractError(
            "D-PCT candidate is missing exact output identity"
        )


def _validate_failure_binding(
    failure: DpctProducerFailureV2,
    *,
    producer_source_view_id: str,
    producer_reference_view_id: str,
) -> None:
    validate_dpct_producer_failure_v2(failure)
    if (
        failure.source_view_id != producer_source_view_id
        or failure.reference_view_id != producer_reference_view_id
    ):
        raise ReferenceMatchContractError(
            "D-PCT failure producer view binding mismatch"
        )


def resolve_dpct_batch_v1(
    *,
    reference: PreparedMatchViewV1,
    sources: Sequence[PreparedMatchViewV1],
    outcomes: Sequence[DpctBatchOutcomeV1],
    adjudications: Sequence[DpctBatchAdjudicationV1] | None = None,
) -> DpctBatchResolutionV1:
    """Resolve a complete batch atomically or require full identity fallback."""

    validate_prepared_match_view(reference)
    if not isinstance(sources, Sequence) or isinstance(
        sources, (str, bytes)
    ):
        raise ReferenceMatchContractError("batch sources must be a sequence")
    if not sources:
        raise ReferenceMatchContractError("batch sources must not be empty")
    if len(sources) > MAX_REFERENCE_MATCH_BATCH_SOURCES:
        raise ReferenceMatchContractError(
            "batch sources exceed the product source limit"
        )
    if not isinstance(outcomes, Sequence) or isinstance(
        outcomes, (str, bytes)
    ):
        raise ReferenceMatchContractError("batch outcomes must be a sequence")
    if len(outcomes) != len(sources):
        raise ReferenceMatchContractError(
            "batch outcome count must match sources"
        )
    producer_reference_view_id = (
        dpct_producer_view_id_for_prepared_v2(reference)
    )
    prepared_sources = tuple(sources)
    producer_source_ids: list[str] = []
    has_failure = False
    for index, (source, outcome) in enumerate(
        zip(prepared_sources, outcomes, strict=True)
    ):
        validate_prepared_match_view(source)
        producer_source_id = dpct_producer_view_id_for_prepared_v2(source)
        producer_source_ids.append(producer_source_id)
        if isinstance(outcome, AdaptedDpctCandidateV2):
            _validate_candidate_binding(
                outcome,
                source=source,
                reference=reference,
                producer_source_view_id=producer_source_id,
                producer_reference_view_id=producer_reference_view_id,
            )
        elif isinstance(outcome, DpctProducerFailureV2):
            _validate_failure_binding(
                outcome,
                producer_source_view_id=producer_source_id,
                producer_reference_view_id=producer_reference_view_id,
            )
            has_failure = True
        else:
            raise ReferenceMatchContractError(
                f"batch outcome {index} type is unsupported"
            )

    if has_failure:
        if adjudications is not None:
            raise ReferenceMatchContractError(
                "producer-failed batch must short-circuit before admission"
            )
        atomic_state = "identity-fallback"
        normalized_adjudications: tuple[DpctBatchAdjudicationV1, ...] = ()
    else:
        if adjudications is None or len(adjudications) != len(sources):
            raise ReferenceMatchContractError(
                "all-candidate batch requires one adjudication per source"
            )
        normalized_adjudications = tuple(adjudications)
        accepted: list[bool] = []
        for index, (outcome, adjudication) in enumerate(
            zip(outcomes, normalized_adjudications, strict=True)
        ):
            if (
                not isinstance(adjudication, tuple)
                or len(adjudication) != 2
            ):
                raise ReferenceMatchContractError(
                    f"batch adjudication {index} must be acceptance/admission"
                )
            acceptance, admission = adjudication
            validate_core_acceptance_decision(acceptance)
            validate_core_candidate_admission_binding(
                admission,
                receipt=outcome.prepared_output.receipt,
                acceptance=acceptance,
            )
            accepted.append(admission.accepted_for_product_guard)
        atomic_state = (
            "pending-product-guard"
            if all(accepted)
            else "identity-fallback"
        )

    source_rows: list[DpctBatchSourceResolutionV1] = []
    for index, (source, outcome) in enumerate(
        zip(prepared_sources, outcomes, strict=True)
    ):
        if isinstance(outcome, DpctProducerFailureV2):
            source_rows.append(
                DpctBatchSourceResolutionV1(
                    source_index=index,
                    source_view_id=source.descriptor.view_id,
                    producer_source_view_id=producer_source_ids[index],
                    producer_status="failed",
                    producer_outcome_id=outcome.diagnostics_id,
                    consumer_transform_id=None,
                    apply_receipt_id=None,
                    acceptance_decision_id=None,
                    admission_id=None,
                    individual_action="identity-fallback",
                )
            )
            continue
        if has_failure:
            acceptance_id = None
            admission_id = None
            individual_action = "not-evaluated"
        else:
            acceptance, admission = normalized_adjudications[index]
            acceptance_id = acceptance.decision_id
            admission_id = admission.admission_id
            individual_action = admission.action
        source_rows.append(
            DpctBatchSourceResolutionV1(
                source_index=index,
                source_view_id=source.descriptor.view_id,
                producer_source_view_id=producer_source_ids[index],
                producer_status="candidate",
                producer_outcome_id=outcome.aliases.result_id,
                consumer_transform_id=outcome.transform.transform_id,
                apply_receipt_id=outcome.prepared_output.receipt.receipt_id,
                acceptance_decision_id=acceptance_id,
                admission_id=admission_id,
                individual_action=individual_action,
            )
        )

    provisional = DpctBatchResolutionV1(
        schema_id=DPCT_BATCH_RESOLUTION_SCHEMA_ID,
        batch_id="0" * 64,
        policy_id=DPCT_BATCH_POLICY_ID,
        compatibility_profile_id=DPCT_COMPATIBILITY_PROFILE_ID,
        producer_commit=DPCT_PINNED_COMMIT,
        reference_view_id=reference.descriptor.view_id,
        producer_reference_view_id=producer_reference_view_id,
        source_count=len(prepared_sources),
        atomic_state=atomic_state,
        required_gates=_REQUIRED_GATES,
        sources=tuple(source_rows),
    )
    result = replace(
        provisional,
        batch_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_dpct_batch_resolution_v1(result)
    return result


def validate_dpct_batch_resolution_v1(
    value: DpctBatchResolutionV1,
) -> None:
    if not isinstance(value, DpctBatchResolutionV1):
        raise ReferenceMatchContractError(
            "D-PCT batch must be DpctBatchResolutionV1"
        )
    if value.schema_id != DPCT_BATCH_RESOLUTION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported D-PCT batch resolution schema"
        )
    _hash(value.batch_id, "batch_id")
    if value.policy_id != DPCT_BATCH_POLICY_ID:
        raise ReferenceMatchContractError("unsupported D-PCT batch policy")
    if value.compatibility_profile_id != DPCT_COMPATIBILITY_PROFILE_ID:
        raise ReferenceMatchContractError(
            "D-PCT batch compatibility profile mismatch"
        )
    if value.producer_commit != DPCT_PINNED_COMMIT:
        raise ReferenceMatchContractError(
            "D-PCT batch producer commit mismatch"
        )
    _hash(value.reference_view_id, "reference_view_id")
    _producer_hash(
        value.producer_reference_view_id,
        "producer_reference_view_id",
    )
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.sources)
    ):
        raise ReferenceMatchContractError(
            "D-PCT batch source_count mismatch"
        )
    if value.atomic_state not in {
        "pending-product-guard",
        "identity-fallback",
    }:
        raise ReferenceMatchContractError(
            "D-PCT batch atomic_state is unsupported"
        )
    if value.required_gates != _REQUIRED_GATES:
        raise ReferenceMatchContractError(
            "D-PCT batch must bind A1/A4/A5"
        )
    has_failure = False
    individual_fallback = False
    for index, source in enumerate(value.sources):
        if not isinstance(source, DpctBatchSourceResolutionV1):
            raise ReferenceMatchContractError(
                "D-PCT batch source row type is invalid"
            )
        if source.source_index != index:
            raise ReferenceMatchContractError(
                "D-PCT batch source indices must be contiguous"
            )
        _hash(source.source_view_id, "source.source_view_id")
        _producer_hash(
            source.producer_source_view_id,
            "source.producer_source_view_id",
        )
        _producer_hash(
            source.producer_outcome_id,
            "source.producer_outcome_id",
        )
        if source.producer_status not in {"candidate", "failed"}:
            raise ReferenceMatchContractError(
                "D-PCT batch producer status is unsupported"
            )
        for label in (
            "consumer_transform_id",
            "apply_receipt_id",
            "acceptance_decision_id",
            "admission_id",
        ):
            _optional_hash(getattr(source, label), f"source.{label}")
        if source.producer_status == "failed":
            has_failure = True
            if (
                source.consumer_transform_id is not None
                or source.apply_receipt_id is not None
                or source.acceptance_decision_id is not None
                or source.admission_id is not None
                or source.individual_action != "identity-fallback"
            ):
                raise ReferenceMatchContractError(
                    "failed D-PCT source must be receipt-free fallback"
                )
        else:
            if (
                source.consumer_transform_id is None
                or source.apply_receipt_id is None
            ):
                raise ReferenceMatchContractError(
                    "candidate D-PCT source must bind transform and receipt"
                )
            if source.individual_action not in {
                "candidate-for-product-guard",
                "identity-fallback",
                "not-evaluated",
            }:
                raise ReferenceMatchContractError(
                    "candidate individual action is unsupported"
                )
            if source.individual_action == "not-evaluated":
                if (
                    source.acceptance_decision_id is not None
                    or source.admission_id is not None
                ):
                    raise ReferenceMatchContractError(
                        "not-evaluated source cannot bind admission"
                    )
            elif (
                source.acceptance_decision_id is None
                or source.admission_id is None
            ):
                raise ReferenceMatchContractError(
                    "evaluated source must bind admission identities"
                )
            if source.individual_action == "identity-fallback":
                individual_fallback = True
    expected = (
        "identity-fallback"
        if has_failure or individual_fallback
        else "pending-product-guard"
    )
    if value.atomic_state != expected:
        raise ReferenceMatchContractError(
            "D-PCT batch atomic state is inconsistent"
        )
    if has_failure and any(
        row.individual_action != "not-evaluated"
        for row in value.sources
        if row.producer_status == "candidate"
    ):
        raise ReferenceMatchContractError(
            "producer failure must short-circuit candidate admission"
        )
    if value.batch_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "D-PCT batch_id does not match canonical payload"
        )


def dpct_batch_resolution_to_json(
    value: DpctBatchResolutionV1,
) -> str:
    validate_dpct_batch_resolution_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def dpct_batch_resolution_from_json(
    encoded: str,
) -> DpctBatchResolutionV1:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "D-PCT batch resolution is not valid JSON"
        ) from exc
    payload = _strict(payload, _RESOLUTION_KEYS, "D-PCT batch resolution")
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ReferenceMatchContractError(
            "D-PCT batch sources must be a non-empty array"
        )
    sources: list[DpctBatchSourceResolutionV1] = []
    for index, raw in enumerate(raw_sources):
        raw = _strict(raw, _SOURCE_KEYS, f"D-PCT batch source {index}")
        try:
            sources.append(DpctBatchSourceResolutionV1(**dict(raw)))
        except TypeError as exc:
            raise ReferenceMatchContractError(
                "D-PCT batch source fields are invalid"
            ) from exc
    try:
        value = DpctBatchResolutionV1(
            schema_id=payload["schema_id"],
            batch_id=payload["batch_id"],
            policy_id=payload["policy_id"],
            compatibility_profile_id=payload[
                "compatibility_profile_id"
            ],
            producer_commit=payload["producer_commit"],
            reference_view_id=payload["reference_view_id"],
            producer_reference_view_id=payload[
                "producer_reference_view_id"
            ],
            source_count=payload["source_count"],
            atomic_state=payload["atomic_state"],
            required_gates=tuple(payload["required_gates"]),
            sources=tuple(sources),
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "D-PCT batch resolution fields are invalid"
        ) from exc
    validate_dpct_batch_resolution_v1(value)
    return value


__all__ = [
    "DPCT_BATCH_POLICY_ID",
    "DPCT_BATCH_RESOLUTION_SCHEMA_ID",
    "DpctBatchResolutionV1",
    "DpctBatchSourceResolutionV1",
    "dpct_batch_resolution_from_json",
    "dpct_batch_resolution_to_json",
    "resolve_dpct_batch_v1",
    "validate_dpct_batch_resolution_v1",
]
