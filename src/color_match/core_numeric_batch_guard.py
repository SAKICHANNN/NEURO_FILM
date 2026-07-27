"""Atomic batch aggregation for exact external-core numeric guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any, Mapping, Sequence

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_numeric_guard import (
    CORE_NUMERIC_GUARD_CLAIM_CEILING,
    CoreNumericGuardDecisionV1,
    validate_core_numeric_guard_decision_v1,
)
from .dpct_batch import (
    DpctBatchResolutionV1,
    validate_dpct_batch_resolution_v1,
)


CORE_NUMERIC_BATCH_GUARD_SCHEMA_ID = (
    "neuro-film.core-numeric-batch-guard.v1"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_BATCH_KEYS = {
    "schema_id",
    "guard_batch_id",
    "upstream_batch_id",
    "upstream_atomic_state",
    "numeric_policy_id",
    "source_count",
    "atomic_state",
    "sources",
    "claim_ceiling",
}
_SOURCE_KEYS = {
    "source_index",
    "source_view_id",
    "apply_receipt_id",
    "admission_id",
    "numeric_decision_id",
    "numeric_accepted_for_transaction",
    "individual_action",
}


@dataclass(frozen=True)
class CoreNumericBatchGuardSourceV1:
    source_index: int
    source_view_id: str
    apply_receipt_id: str | None
    admission_id: str | None
    numeric_decision_id: str | None
    numeric_accepted_for_transaction: bool | None
    individual_action: str


@dataclass(frozen=True)
class CoreNumericBatchGuardV1:
    """All-or-nothing numeric guard; never a delivered result."""

    schema_id: str
    guard_batch_id: str
    upstream_batch_id: str
    upstream_atomic_state: str
    numeric_policy_id: str | None
    source_count: int
    atomic_state: str
    sources: tuple[CoreNumericBatchGuardSourceV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [asdict(source) for source in self.sources]
        return payload


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


def _optional_hash(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _hash(value, label)


def _identity_payload(value: CoreNumericBatchGuardV1) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("guard_batch_id")
    return payload


def guard_core_numeric_batch_v1(
    *,
    batch: DpctBatchResolutionV1,
    decisions: Sequence[CoreNumericGuardDecisionV1] | None = None,
) -> CoreNumericBatchGuardV1:
    """Aggregate exact per-source decisions without partial eligibility."""

    validate_dpct_batch_resolution_v1(batch)
    if batch.atomic_state == "identity-fallback":
        if decisions is not None:
            raise ReferenceMatchContractError(
                "upstream fallback batch cannot accept numeric decisions"
            )
        policy_id = None
        rows = tuple(
            CoreNumericBatchGuardSourceV1(
                source_index=row.source_index,
                source_view_id=row.source_view_id,
                apply_receipt_id=row.apply_receipt_id,
                admission_id=row.admission_id,
                numeric_decision_id=None,
                numeric_accepted_for_transaction=None,
                individual_action="not-evaluated",
            )
            for row in batch.sources
        )
        atomic_state = "identity-fallback"
    else:
        if (
            decisions is None
            or not isinstance(decisions, Sequence)
            or isinstance(decisions, (str, bytes))
            or len(decisions) != batch.source_count
        ):
            raise ReferenceMatchContractError(
                "pending batch requires one numeric decision per source"
            )
        normalized = tuple(decisions)
        policy_ids: set[str] = set()
        built_rows: list[CoreNumericBatchGuardSourceV1] = []
        for index, (source, decision) in enumerate(
            zip(batch.sources, normalized, strict=True)
        ):
            validate_core_numeric_guard_decision_v1(decision)
            if (
                source.source_index != index
                or decision.source_view_id != source.source_view_id
            ):
                raise ReferenceMatchContractError(
                    f"numeric decision {index} source binding mismatch"
                )
            if (
                decision.receipt_id != source.apply_receipt_id
                or decision.admission_id != source.admission_id
                or not decision.admission_accepted_for_product_guard
            ):
                raise ReferenceMatchContractError(
                    f"numeric decision {index} does not bind batch source"
                )
            policy_ids.add(decision.policy_id)
            built_rows.append(
                CoreNumericBatchGuardSourceV1(
                    source_index=index,
                    source_view_id=source.source_view_id,
                    apply_receipt_id=source.apply_receipt_id,
                    admission_id=source.admission_id,
                    numeric_decision_id=decision.decision_id,
                    numeric_accepted_for_transaction=(
                        decision.accepted_for_transaction
                    ),
                    individual_action=decision.action,
                )
            )
        if len(policy_ids) != 1:
            raise ReferenceMatchContractError(
                "numeric batch decisions must share one policy"
            )
        policy_id = next(iter(policy_ids))
        rows = tuple(built_rows)
        atomic_state = (
            "eligible-for-transaction"
            if all(
                row.numeric_accepted_for_transaction is True
                for row in rows
            )
            else "identity-fallback"
        )

    provisional = CoreNumericBatchGuardV1(
        schema_id=CORE_NUMERIC_BATCH_GUARD_SCHEMA_ID,
        guard_batch_id="0" * 64,
        upstream_batch_id=batch.batch_id,
        upstream_atomic_state=batch.atomic_state,
        numeric_policy_id=policy_id,
        source_count=batch.source_count,
        atomic_state=atomic_state,
        sources=rows,
        claim_ceiling=CORE_NUMERIC_GUARD_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        guard_batch_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_core_numeric_batch_guard_v1(result)
    return result


def validate_core_numeric_batch_guard_v1(
    value: CoreNumericBatchGuardV1,
) -> None:
    if not isinstance(value, CoreNumericBatchGuardV1):
        raise ReferenceMatchContractError(
            "numeric batch guard must be CoreNumericBatchGuardV1"
        )
    if value.schema_id != CORE_NUMERIC_BATCH_GUARD_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported numeric batch guard schema"
        )
    _hash(value.guard_batch_id, "guard_batch_id")
    _hash(value.upstream_batch_id, "upstream_batch_id")
    if value.upstream_atomic_state not in {
        "pending-product-guard",
        "identity-fallback",
    }:
        raise ReferenceMatchContractError(
            "numeric batch upstream state is unsupported"
        )
    _optional_hash(value.numeric_policy_id, "numeric_policy_id")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count != len(value.sources)
    ):
        raise ReferenceMatchContractError(
            "numeric batch source_count mismatch"
        )
    if value.atomic_state not in {
        "eligible-for-transaction",
        "identity-fallback",
    }:
        raise ReferenceMatchContractError(
            "numeric batch atomic state is unsupported"
        )
    if value.claim_ceiling != CORE_NUMERIC_GUARD_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "numeric batch claim ceiling mismatch"
        )
    accepted: list[bool] = []
    for index, source in enumerate(value.sources):
        if not isinstance(source, CoreNumericBatchGuardSourceV1):
            raise ReferenceMatchContractError(
                "numeric batch source row type is invalid"
            )
        if source.source_index != index:
            raise ReferenceMatchContractError(
                "numeric batch source indices must be contiguous"
            )
        _hash(source.source_view_id, "source.source_view_id")
        _optional_hash(source.apply_receipt_id, "source.apply_receipt_id")
        _optional_hash(source.admission_id, "source.admission_id")
        _optional_hash(
            source.numeric_decision_id,
            "source.numeric_decision_id",
        )
        if (
            source.numeric_accepted_for_transaction is not None
            and not isinstance(
                source.numeric_accepted_for_transaction, bool
            )
        ):
            raise ReferenceMatchContractError(
                "numeric batch source acceptance must be boolean or null"
            )
        if source.individual_action not in {
            "eligible-for-transaction",
            "identity-fallback",
            "not-evaluated",
        }:
            raise ReferenceMatchContractError(
                "numeric batch source action is unsupported"
            )
        if value.upstream_atomic_state == "identity-fallback":
            if (
                source.numeric_decision_id is not None
                or source.numeric_accepted_for_transaction is not None
                or source.individual_action != "not-evaluated"
            ):
                raise ReferenceMatchContractError(
                    "upstream fallback sources must remain not-evaluated"
                )
        else:
            if (
                source.apply_receipt_id is None
                or source.admission_id is None
                or source.numeric_decision_id is None
                or source.numeric_accepted_for_transaction is None
                or source.individual_action
                != (
                    "eligible-for-transaction"
                    if source.numeric_accepted_for_transaction
                    else "identity-fallback"
                )
            ):
                raise ReferenceMatchContractError(
                    "pending batch source decision state is inconsistent"
                )
            accepted.append(source.numeric_accepted_for_transaction)
    if value.upstream_atomic_state == "identity-fallback":
        expected_state = "identity-fallback"
        if value.numeric_policy_id is not None:
            raise ReferenceMatchContractError(
                "upstream fallback cannot bind a numeric policy"
            )
    else:
        if value.numeric_policy_id is None:
            raise ReferenceMatchContractError(
                "evaluated numeric batch must bind one policy"
            )
        expected_state = (
            "eligible-for-transaction"
            if all(accepted)
            else "identity-fallback"
        )
    if value.atomic_state != expected_state:
        raise ReferenceMatchContractError(
            "numeric batch atomic state is inconsistent"
        )
    if value.guard_batch_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "numeric guard_batch_id mismatch"
        )


def core_numeric_batch_guard_to_json(
    value: CoreNumericBatchGuardV1,
) -> str:
    validate_core_numeric_batch_guard_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def core_numeric_batch_guard_from_json(
    encoded: str,
) -> CoreNumericBatchGuardV1:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "numeric batch guard is not valid JSON"
        ) from exc
    payload = _strict(payload, _BATCH_KEYS, "numeric batch guard")
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ReferenceMatchContractError(
            "numeric batch sources must be a non-empty array"
        )
    sources: list[CoreNumericBatchGuardSourceV1] = []
    for index, raw in enumerate(raw_sources):
        raw = _strict(raw, _SOURCE_KEYS, f"numeric batch source {index}")
        try:
            sources.append(CoreNumericBatchGuardSourceV1(**dict(raw)))
        except TypeError as exc:
            raise ReferenceMatchContractError(
                "numeric batch source fields are invalid"
            ) from exc
    try:
        value = CoreNumericBatchGuardV1(
            schema_id=payload["schema_id"],
            guard_batch_id=payload["guard_batch_id"],
            upstream_batch_id=payload["upstream_batch_id"],
            upstream_atomic_state=payload["upstream_atomic_state"],
            numeric_policy_id=payload["numeric_policy_id"],
            source_count=payload["source_count"],
            atomic_state=payload["atomic_state"],
            sources=tuple(sources),
            claim_ceiling=payload["claim_ceiling"],
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "numeric batch guard fields are invalid"
        ) from exc
    validate_core_numeric_batch_guard_v1(value)
    return value


__all__ = [
    "CORE_NUMERIC_BATCH_GUARD_SCHEMA_ID",
    "CoreNumericBatchGuardSourceV1",
    "CoreNumericBatchGuardV1",
    "core_numeric_batch_guard_from_json",
    "core_numeric_batch_guard_to_json",
    "guard_core_numeric_batch_v1",
    "validate_core_numeric_batch_guard_v1",
]
