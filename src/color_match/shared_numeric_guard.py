"""Atomic numeric guard for exact shared-operator batch applications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_adapter import PreparedMatchViewV1, validate_prepared_match_view
from .core_numeric_guard import (
    CORE_NUMERIC_GUARD_CLAIM_CEILING,
    CoreNumericGuardPolicyV1,
    core_numeric_guard_policy_id,
    validate_core_numeric_guard_policy_v1,
)
from .shared_operator_batch import (
    PreparedSharedOperatorApplyV1,
    SharedOperatorApplyReceiptV1,
    SharedOperatorBatchV1,
    validate_prepared_shared_operator_apply_v1,
    validate_shared_operator_batch_v1,
)


SHARED_NUMERIC_FACTS_SCHEMA_ID = (
    "neuro-film.shared-apply-numeric-facts.v1"
)
SHARED_NUMERIC_DECISION_SCHEMA_ID = (
    "neuro-film.shared-apply-numeric-decision.v1"
)
SHARED_NUMERIC_BATCH_SCHEMA_ID = (
    "neuro-film.shared-numeric-batch-guard.v1"
)
_PRODUCER_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_FACT_KEYS = {
    "schema_id",
    "facts_id",
    "receipt_id",
    "producer_diagnostics_id",
    "all_finite",
    "output_minimum",
    "output_maximum",
    "out_of_gamut_fraction",
    "clipping_fraction",
    "projected_fraction",
}
_DECISION_KEYS = {
    "schema_id",
    "decision_id",
    "upstream_batch_id",
    "operator_id",
    "policy_id",
    "source_index",
    "source_view_id",
    "receipt_id",
    "output_view_id",
    "facts",
    "accepted_for_transaction",
    "action",
    "reasons",
    "new_boundary_fraction",
}
_BATCH_KEYS = {
    "schema_id",
    "guard_batch_id",
    "upstream_batch_id",
    "operator_id",
    "policy",
    "policy_id",
    "source_count",
    "atomic_state",
    "decisions",
    "claim_ceiling",
}
_POLICY_KEYS = {
    "schema_id",
    "max_out_of_gamut_fraction",
    "max_clipping_fraction",
    "max_projected_fraction",
    "max_new_boundary_fraction",
    "boundary_epsilon",
}


@dataclass(frozen=True)
class SharedApplyNumericFactsV1:
    schema_id: str
    facts_id: str
    receipt_id: str
    producer_diagnostics_id: str
    all_finite: bool
    output_minimum: float
    output_maximum: float
    out_of_gamut_fraction: float
    clipping_fraction: float
    projected_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SharedApplyNumericDecisionV1:
    schema_id: str
    decision_id: str
    upstream_batch_id: str
    operator_id: str
    policy_id: str
    source_index: int
    source_view_id: str
    receipt_id: str
    output_view_id: str
    facts: SharedApplyNumericFactsV1
    accepted_for_transaction: bool
    action: str
    reasons: tuple[str, ...]
    new_boundary_fraction: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["facts"] = self.facts.to_dict()
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class SharedNumericBatchGuardV1:
    schema_id: str
    guard_batch_id: str
    upstream_batch_id: str
    operator_id: str
    policy: CoreNumericGuardPolicyV1
    policy_id: str
    source_count: int
    atomic_state: str
    decisions: tuple[SharedApplyNumericDecisionV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy"] = self.policy.to_dict()
        payload["decisions"] = [
            decision.to_dict() for decision in self.decisions
        ]
        return payload


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the shared numeric contract"
        )
    return value


def _hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
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


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ReferenceMatchContractError(f"{label} must be finite")
    return float(value)


def _fraction(value: Any, label: str) -> float:
    number = _finite(value, label)
    if not 0.0 <= number <= 1.0:
        raise ReferenceMatchContractError(
            f"{label} must be within [0, 1]"
        )
    return number


def _identity_payload(value: Any, identity_key: str) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop(identity_key)
    return payload


def make_shared_apply_numeric_facts_v1(
    *,
    prepared: PreparedSharedOperatorApplyV1,
    producer_diagnostics_id: str,
    all_finite: bool,
    output_minimum: float,
    output_maximum: float,
    out_of_gamut_fraction: float,
    clipping_fraction: float,
    projected_fraction: float,
) -> SharedApplyNumericFactsV1:
    """Bind producer facts and independently verify exact output extrema."""

    validate_prepared_shared_operator_apply_v1(prepared)
    receipt = prepared.receipt
    diagnostics_id = _producer_hash(
        producer_diagnostics_id, "producer_diagnostics_id"
    )
    if diagnostics_id != receipt.diagnostics_id:
        raise ReferenceMatchContractError(
            "shared numeric diagnostics/receipt mismatch"
        )
    if all_finite is not True:
        raise ReferenceMatchContractError(
            "shared candidate facts require all_finite true"
        )
    minimum = _finite(output_minimum, "output_minimum")
    maximum = _finite(output_maximum, "output_maximum")
    actual_minimum = float(np.min(prepared.pixels))
    actual_maximum = float(np.max(prepared.pixels))
    if minimum != actual_minimum or maximum != actual_maximum:
        raise ReferenceMatchContractError(
            "shared numeric output extrema mismatch"
        )
    provisional = SharedApplyNumericFactsV1(
        schema_id=SHARED_NUMERIC_FACTS_SCHEMA_ID,
        facts_id="0" * 64,
        receipt_id=receipt.receipt_id,
        producer_diagnostics_id=diagnostics_id,
        all_finite=True,
        output_minimum=minimum,
        output_maximum=maximum,
        out_of_gamut_fraction=_fraction(
            out_of_gamut_fraction, "out_of_gamut_fraction"
        ),
        clipping_fraction=_fraction(
            clipping_fraction, "clipping_fraction"
        ),
        projected_fraction=_fraction(
            projected_fraction, "projected_fraction"
        ),
    )
    result = replace(
        provisional,
        facts_id=canonical_sha256(
            _identity_payload(provisional, "facts_id")
        ),
    )
    validate_shared_apply_numeric_facts_v1(result)
    return result


def validate_shared_apply_numeric_facts_v1(
    value: SharedApplyNumericFactsV1,
) -> None:
    if not isinstance(value, SharedApplyNumericFactsV1):
        raise ReferenceMatchContractError(
            "shared numeric facts type is invalid"
        )
    if value.schema_id != SHARED_NUMERIC_FACTS_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared numeric facts schema is invalid"
        )
    _hash(value.facts_id, "facts_id")
    _hash(value.receipt_id, "receipt_id")
    _producer_hash(
        value.producer_diagnostics_id,
        "producer_diagnostics_id",
    )
    if value.all_finite is not True:
        raise ReferenceMatchContractError(
            "shared numeric facts must be all finite"
        )
    minimum = _finite(value.output_minimum, "output_minimum")
    maximum = _finite(value.output_maximum, "output_maximum")
    if minimum > maximum:
        raise ReferenceMatchContractError(
            "shared numeric output extrema are reversed"
        )
    for key in (
        "out_of_gamut_fraction",
        "clipping_fraction",
        "projected_fraction",
    ):
        _fraction(getattr(value, key), key)
    if value.facts_id != canonical_sha256(
        _identity_payload(value, "facts_id")
    ):
        raise ReferenceMatchContractError(
            "shared numeric facts identity mismatch"
        )


def _new_boundary_fraction(
    source: np.ndarray,
    output: np.ndarray,
    epsilon: float,
) -> float:
    source_boundary = np.any(
        (source <= epsilon) | (source >= 1.0 - epsilon),
        axis=-1,
    )
    output_boundary = np.any(
        (output <= epsilon) | (output >= 1.0 - epsilon),
        axis=-1,
    )
    return float(np.mean(output_boundary & ~source_boundary, dtype=np.float64))


def _decision_reasons(
    facts: SharedApplyNumericFactsV1,
    new_boundary: float,
    policy: CoreNumericGuardPolicyV1,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if facts.out_of_gamut_fraction > policy.max_out_of_gamut_fraction:
        reasons.append("out-of-gamut-fraction")
    if facts.clipping_fraction > policy.max_clipping_fraction:
        reasons.append("clipping-fraction")
    if facts.projected_fraction > policy.max_projected_fraction:
        reasons.append("projected-fraction")
    if new_boundary > policy.max_new_boundary_fraction:
        reasons.append("new-boundary-fraction")
    return tuple(sorted(reasons))


def guard_shared_numeric_batch_v1(
    *,
    batch: SharedOperatorBatchV1,
    sources: Sequence[PreparedMatchViewV1],
    applies: Sequence[PreparedSharedOperatorApplyV1],
    facts: Sequence[SharedApplyNumericFactsV1],
    policy: CoreNumericGuardPolicyV1 | None = None,
) -> SharedNumericBatchGuardV1:
    """Guard every exact apply and fall back atomically on any failure."""

    validate_shared_operator_batch_v1(batch)
    resolved = policy or CoreNumericGuardPolicyV1()
    validate_core_numeric_guard_policy_v1(resolved)
    if (
        not isinstance(sources, Sequence)
        or isinstance(sources, (str, bytes))
        or not isinstance(applies, Sequence)
        or isinstance(applies, (str, bytes))
        or not isinstance(facts, Sequence)
        or isinstance(facts, (str, bytes))
        or len(sources) != batch.source_count
        or len(applies) != batch.source_count
        or len(facts) != batch.source_count
    ):
        raise ReferenceMatchContractError(
            "shared numeric batch inventory is incomplete"
        )
    policy_id = core_numeric_guard_policy_id(resolved)
    decisions: list[SharedApplyNumericDecisionV1] = []
    for index, (source, prepared, item) in enumerate(
        zip(sources, applies, facts, strict=True)
    ):
        validate_prepared_match_view(source)
        validate_prepared_shared_operator_apply_v1(prepared)
        validate_shared_apply_numeric_facts_v1(item)
        receipt = prepared.receipt
        if (
            receipt != batch.sources[index]
            or receipt.source_index != index
            or receipt.source_view_id != source.descriptor.view_id
            or item.receipt_id != receipt.receipt_id
            or item.producer_diagnostics_id != receipt.diagnostics_id
        ):
            raise ReferenceMatchContractError(
                "shared numeric source binding mismatch"
            )
        if (
            item.output_minimum != float(np.min(prepared.pixels))
            or item.output_maximum != float(np.max(prepared.pixels))
        ):
            raise ReferenceMatchContractError(
                "shared numeric live output extrema mismatch"
            )
        new_boundary = _new_boundary_fraction(
            source.pixels,
            prepared.pixels,
            float(resolved.boundary_epsilon),
        )
        reasons = _decision_reasons(item, new_boundary, resolved)
        accepted = not reasons
        provisional = SharedApplyNumericDecisionV1(
            schema_id=SHARED_NUMERIC_DECISION_SCHEMA_ID,
            decision_id="0" * 64,
            upstream_batch_id=batch.batch_id,
            operator_id=batch.operator.operator_id,
            policy_id=policy_id,
            source_index=index,
            source_view_id=source.descriptor.view_id,
            receipt_id=receipt.receipt_id,
            output_view_id=receipt.output_view.view_id,
            facts=item,
            accepted_for_transaction=accepted,
            action=(
                "eligible-for-transaction"
                if accepted
                else "identity-fallback"
            ),
            reasons=reasons,
            new_boundary_fraction=new_boundary,
        )
        decisions.append(
            replace(
                provisional,
                decision_id=canonical_sha256(
                    _identity_payload(provisional, "decision_id")
                ),
            )
        )
    atomic_state = (
        "eligible-for-transaction"
        if all(item.accepted_for_transaction for item in decisions)
        else "identity-fallback"
    )
    provisional_batch = SharedNumericBatchGuardV1(
        schema_id=SHARED_NUMERIC_BATCH_SCHEMA_ID,
        guard_batch_id="0" * 64,
        upstream_batch_id=batch.batch_id,
        operator_id=batch.operator.operator_id,
        policy=resolved,
        policy_id=policy_id,
        source_count=batch.source_count,
        atomic_state=atomic_state,
        decisions=tuple(decisions),
        claim_ceiling=CORE_NUMERIC_GUARD_CLAIM_CEILING,
    )
    result = replace(
        provisional_batch,
        guard_batch_id=canonical_sha256(
            _identity_payload(provisional_batch, "guard_batch_id")
        ),
    )
    validate_shared_numeric_batch_guard_v1(result)
    return result


def validate_shared_apply_numeric_decision_v1(
    value: SharedApplyNumericDecisionV1,
    *,
    policy: CoreNumericGuardPolicyV1,
) -> None:
    if not isinstance(value, SharedApplyNumericDecisionV1):
        raise ReferenceMatchContractError(
            "shared numeric decision type is invalid"
        )
    if value.schema_id != SHARED_NUMERIC_DECISION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared numeric decision schema is invalid"
        )
    for key in (
        "decision_id",
        "upstream_batch_id",
        "operator_id",
        "policy_id",
        "source_view_id",
        "receipt_id",
        "output_view_id",
    ):
        _hash(getattr(value, key), key)
    if value.policy_id != core_numeric_guard_policy_id(policy):
        raise ReferenceMatchContractError(
            "shared numeric decision policy mismatch"
        )
    if (
        isinstance(value.source_index, bool)
        or not isinstance(value.source_index, int)
        or value.source_index < 0
    ):
        raise ReferenceMatchContractError(
            "shared numeric decision source_index is invalid"
        )
    validate_shared_apply_numeric_facts_v1(value.facts)
    if value.facts.receipt_id != value.receipt_id:
        raise ReferenceMatchContractError(
            "shared numeric decision facts/receipt mismatch"
        )
    new_boundary = _fraction(
        value.new_boundary_fraction, "new_boundary_fraction"
    )
    expected_reasons = _decision_reasons(value.facts, new_boundary, policy)
    expected_accepted = not expected_reasons
    expected_action = (
        "eligible-for-transaction"
        if expected_accepted
        else "identity-fallback"
    )
    if (
        value.reasons != expected_reasons
        or value.accepted_for_transaction != expected_accepted
        or value.action != expected_action
    ):
        raise ReferenceMatchContractError(
            "shared numeric decision state is inconsistent"
        )
    if value.decision_id != canonical_sha256(
        _identity_payload(value, "decision_id")
    ):
        raise ReferenceMatchContractError(
            "shared numeric decision identity mismatch"
        )


def validate_shared_numeric_batch_guard_v1(
    value: SharedNumericBatchGuardV1,
) -> None:
    if not isinstance(value, SharedNumericBatchGuardV1):
        raise ReferenceMatchContractError(
            "shared numeric batch type is invalid"
        )
    if value.schema_id != SHARED_NUMERIC_BATCH_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared numeric batch schema is invalid"
        )
    _hash(value.guard_batch_id, "guard_batch_id")
    _hash(value.upstream_batch_id, "upstream_batch_id")
    _hash(value.operator_id, "operator_id")
    validate_core_numeric_guard_policy_v1(value.policy)
    if value.policy_id != core_numeric_guard_policy_id(value.policy):
        raise ReferenceMatchContractError(
            "shared numeric batch policy mismatch"
        )
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count != len(value.decisions)
    ):
        raise ReferenceMatchContractError(
            "shared numeric batch source_count is invalid"
        )
    accepted: list[bool] = []
    for index, decision in enumerate(value.decisions):
        validate_shared_apply_numeric_decision_v1(
            decision, policy=value.policy
        )
        if (
            decision.source_index != index
            or decision.upstream_batch_id != value.upstream_batch_id
            or decision.operator_id != value.operator_id
        ):
            raise ReferenceMatchContractError(
                "shared numeric batch decision binding mismatch"
            )
        accepted.append(decision.accepted_for_transaction)
    expected_state = (
        "eligible-for-transaction"
        if all(accepted)
        else "identity-fallback"
    )
    if value.atomic_state != expected_state:
        raise ReferenceMatchContractError(
            "shared numeric batch atomic state is inconsistent"
        )
    if value.claim_ceiling != CORE_NUMERIC_GUARD_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "shared numeric batch claim ceiling mismatch"
        )
    if value.guard_batch_id != canonical_sha256(
        _identity_payload(value, "guard_batch_id")
    ):
        raise ReferenceMatchContractError(
            "shared numeric batch identity mismatch"
        )


def shared_numeric_batch_guard_to_json(
    value: SharedNumericBatchGuardV1,
) -> str:
    validate_shared_numeric_batch_guard_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def shared_numeric_batch_guard_from_json(
    encoded: str,
) -> SharedNumericBatchGuardV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "shared numeric batch is not valid JSON"
        ) from exc
    payload = _strict(payload, _BATCH_KEYS, "shared numeric batch")
    policy_payload = _strict(
        payload["policy"], _POLICY_KEYS, "shared numeric policy"
    )
    try:
        policy = CoreNumericGuardPolicyV1(**dict(policy_payload))
    except TypeError as exc:
        raise ReferenceMatchContractError(
            "shared numeric policy fields are invalid"
        ) from exc
    if not isinstance(payload["decisions"], list):
        raise ReferenceMatchContractError(
            "shared numeric decisions must be an array"
        )
    decisions: list[SharedApplyNumericDecisionV1] = []
    for index, raw in enumerate(payload["decisions"]):
        raw = dict(
            _strict(raw, _DECISION_KEYS, f"shared decision {index}")
        )
        facts_payload = _strict(
            raw["facts"], _FACT_KEYS, f"shared facts {index}"
        )
        try:
            raw["facts"] = SharedApplyNumericFactsV1(**dict(facts_payload))
            raw["reasons"] = tuple(raw["reasons"])
            decisions.append(SharedApplyNumericDecisionV1(**raw))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "shared numeric decision fields are invalid"
            ) from exc
    try:
        result = SharedNumericBatchGuardV1(
            schema_id=payload["schema_id"],
            guard_batch_id=payload["guard_batch_id"],
            upstream_batch_id=payload["upstream_batch_id"],
            operator_id=payload["operator_id"],
            policy=policy,
            policy_id=payload["policy_id"],
            source_count=payload["source_count"],
            atomic_state=payload["atomic_state"],
            decisions=tuple(decisions),
            claim_ceiling=payload["claim_ceiling"],
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "shared numeric batch fields are invalid"
        ) from exc
    validate_shared_numeric_batch_guard_v1(result)
    return result


__all__ = [
    "SHARED_NUMERIC_BATCH_SCHEMA_ID",
    "SHARED_NUMERIC_DECISION_SCHEMA_ID",
    "SHARED_NUMERIC_FACTS_SCHEMA_ID",
    "SharedApplyNumericDecisionV1",
    "SharedApplyNumericFactsV1",
    "SharedNumericBatchGuardV1",
    "guard_shared_numeric_batch_v1",
    "make_shared_apply_numeric_facts_v1",
    "shared_numeric_batch_guard_from_json",
    "shared_numeric_batch_guard_to_json",
    "validate_shared_apply_numeric_decision_v1",
    "validate_shared_apply_numeric_facts_v1",
    "validate_shared_numeric_batch_guard_v1",
]
