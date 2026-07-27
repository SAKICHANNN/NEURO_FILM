"""Fail-closed product intake for external-core colour candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from typing import Any, Mapping

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_contracts import (
    CapabilitiesV1,
    DiagnosticsV1,
    MatchViewV1,
    TransformBundleV1,
    validate_core_binding,
)
from .promotion import PromotionDecision


CORE_ACCEPTANCE_SCHEMA_ID = "neuro-film.reference-core-acceptance.v1"
CORE_ACCEPTANCE_POLICY_ID = "reference-core-acceptance.v1"
_REQUIRED_GATES = ("A1", "A4", "A5")
_KEYS = {
    "schema_id",
    "decision_id",
    "policy_id",
    "transform_id",
    "accepted_for_product_guard",
    "action",
    "reasons",
    "core_status",
    "core_fallback_reason",
    "promotion_status",
    "promotion_reasons",
    "required_gates",
    "research_baseline_override",
}


@dataclass(frozen=True)
class CoreAcceptanceDecisionV1:
    """External-core intake result; never a final pixel delivery decision."""

    schema_id: str
    decision_id: str
    policy_id: str
    transform_id: str
    accepted_for_product_guard: bool
    action: str
    reasons: tuple[str, ...]
    core_status: str
    core_fallback_reason: str | None
    promotion_status: str
    promotion_reasons: tuple[str, ...]
    required_gates: tuple[str, ...]
    research_baseline_override: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["promotion_reasons"] = list(self.promotion_reasons)
        payload["required_gates"] = list(self.required_gates)
        return payload


def _strict_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ReferenceMatchContractError(
            f"{label} keys mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceMatchContractError(
            f"{label} must be a non-empty string"
        )
    return value


def _sorted_unique(
    value: Any,
    label: str,
    *,
    nonempty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReferenceMatchContractError(f"{label} must be an array")
    result = tuple(_nonempty(item, f"{label}[]") for item in value)
    if nonempty and not result:
        raise ReferenceMatchContractError(f"{label} must not be empty")
    if result != tuple(sorted(set(result))):
        raise ReferenceMatchContractError(
            f"{label} must be sorted and unique"
        )
    return result


def _validate_promotion(value: PromotionDecision) -> None:
    if not isinstance(value, PromotionDecision):
        raise ReferenceMatchContractError(
            "promotion must be PromotionDecision"
        )
    if value.status not in {
        "rejected",
        "eligible-for-visual-review",
        "promoted",
    }:
        raise ReferenceMatchContractError(
            "unsupported promotion status"
        )
    reasons = tuple(value.reasons)
    if any(not isinstance(reason, str) or not reason for reason in reasons):
        raise ReferenceMatchContractError(
            "promotion reasons must be non-empty strings"
        )
    if value.status == "promoted" and reasons:
        raise ReferenceMatchContractError(
            "promoted decision must not contain reasons"
        )
    if value.status != "promoted" and not reasons:
        raise ReferenceMatchContractError(
            "unpromoted decision must contain reasons"
        )


def _identity_payload(
    value: CoreAcceptanceDecisionV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("decision_id")
    return payload


def adjudicate_core_acceptance(
    *,
    source: MatchViewV1,
    reference: MatchViewV1,
    transform: TransformBundleV1,
    capabilities: CapabilitiesV1,
    diagnostics: DiagnosticsV1,
    promotion: PromotionDecision,
    allow_research_baseline: bool = False,
) -> CoreAcceptanceDecisionV1:
    """Admit a candidate to the pixel guard or require identity fallback."""

    if not isinstance(allow_research_baseline, bool):
        raise ReferenceMatchContractError(
            "allow_research_baseline must be boolean"
        )
    validate_core_binding(
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
    )
    _validate_promotion(promotion)
    reasons: list[str] = []
    if diagnostics.status != "ok":
        reasons.append(f"core-status:{diagnostics.status}")
        reasons.append(
            f"core-fallback:{diagnostics.fallback_reason}"
        )
    if promotion.status != "promoted" and not allow_research_baseline:
        reasons.append("algorithm-not-promoted")
    normalized_reasons = tuple(sorted(set(reasons)))
    accepted = not normalized_reasons
    provisional = CoreAcceptanceDecisionV1(
        schema_id=CORE_ACCEPTANCE_SCHEMA_ID,
        decision_id="0" * 64,
        policy_id=CORE_ACCEPTANCE_POLICY_ID,
        transform_id=transform.transform_id,
        accepted_for_product_guard=accepted,
        action=(
            "candidate-for-product-guard"
            if accepted
            else "identity-fallback"
        ),
        reasons=normalized_reasons,
        core_status=diagnostics.status,
        core_fallback_reason=diagnostics.fallback_reason,
        promotion_status=promotion.status,
        promotion_reasons=tuple(sorted(set(promotion.reasons))),
        required_gates=_REQUIRED_GATES,
        research_baseline_override=allow_research_baseline,
    )
    result = replace(
        provisional,
        decision_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_core_acceptance_decision(result)
    return result


def validate_core_acceptance_decision(
    value: CoreAcceptanceDecisionV1,
) -> None:
    if not isinstance(value, CoreAcceptanceDecisionV1):
        raise ReferenceMatchContractError(
            "core acceptance must be CoreAcceptanceDecisionV1"
        )
    if value.schema_id != CORE_ACCEPTANCE_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported core-acceptance schema"
        )
    _sha256(value.decision_id, "decision_id")
    if value.policy_id != CORE_ACCEPTANCE_POLICY_ID:
        raise ReferenceMatchContractError(
            "unsupported core-acceptance policy"
        )
    _sha256(value.transform_id, "transform_id")
    if not isinstance(value.accepted_for_product_guard, bool):
        raise ReferenceMatchContractError(
            "accepted_for_product_guard must be boolean"
        )
    if value.action not in {
        "candidate-for-product-guard",
        "identity-fallback",
    }:
        raise ReferenceMatchContractError(
            "unsupported core-acceptance action"
        )
    reasons = _sorted_unique(value.reasons, "reasons", nonempty=False)
    promotion_reasons = _sorted_unique(
        value.promotion_reasons,
        "promotion_reasons",
        nonempty=value.promotion_status != "promoted",
    )
    if value.required_gates != _REQUIRED_GATES:
        raise ReferenceMatchContractError(
            "core acceptance must bind A1/A4/A5"
        )
    if not isinstance(value.research_baseline_override, bool):
        raise ReferenceMatchContractError(
            "research_baseline_override must be boolean"
        )
    if value.core_status not in {
        "ok",
        "identity-fallback",
        "unsupported",
        "invalid",
    }:
        raise ReferenceMatchContractError("unsupported core status")
    if value.core_status == "ok":
        if value.core_fallback_reason is not None:
            raise ReferenceMatchContractError(
                "ok core status must not have a fallback reason"
            )
    else:
        _nonempty(value.core_fallback_reason, "core_fallback_reason")
        if f"core-status:{value.core_status}" not in reasons:
            raise ReferenceMatchContractError(
                "non-ok core status must fail closed"
            )
    if value.promotion_status not in {
        "rejected",
        "eligible-for-visual-review",
        "promoted",
    }:
        raise ReferenceMatchContractError(
            "unsupported promotion status"
        )
    if value.promotion_status == "promoted" and promotion_reasons:
        raise ReferenceMatchContractError(
            "promoted acceptance must not contain promotion reasons"
        )
    expected_accepted = (
        value.core_status == "ok"
        and (
            value.promotion_status == "promoted"
            or value.research_baseline_override
        )
    )
    if value.accepted_for_product_guard != expected_accepted:
        raise ReferenceMatchContractError(
            "core acceptance state is inconsistent"
        )
    expected_action = (
        "candidate-for-product-guard"
        if expected_accepted
        else "identity-fallback"
    )
    if value.action != expected_action:
        raise ReferenceMatchContractError(
            "core acceptance action is inconsistent"
        )
    if expected_accepted and reasons:
        raise ReferenceMatchContractError(
            "accepted core candidate must not contain rejection reasons"
        )
    if not expected_accepted and not reasons:
        raise ReferenceMatchContractError(
            "identity fallback must contain rejection reasons"
        )
    if (
        value.promotion_status != "promoted"
        and not value.research_baseline_override
        and "algorithm-not-promoted" not in reasons
    ):
        raise ReferenceMatchContractError(
            "unpromoted algorithm must fail closed"
        )
    if value.decision_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "decision_id does not match canonical payload"
        )


def core_acceptance_to_json(value: CoreAcceptanceDecisionV1) -> str:
    validate_core_acceptance_decision(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def core_acceptance_from_json(encoded: str) -> CoreAcceptanceDecisionV1:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "core acceptance is not valid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReferenceMatchContractError(
            "core acceptance must be an object"
        )
    _strict_keys(payload, _KEYS, "core acceptance")
    converted = dict(payload)
    for key in ("reasons", "promotion_reasons", "required_gates"):
        converted[key] = tuple(converted[key])
    try:
        result = CoreAcceptanceDecisionV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "core acceptance contains invalid fields"
        ) from exc
    validate_core_acceptance_decision(result)
    return result


__all__ = [
    "CORE_ACCEPTANCE_POLICY_ID",
    "CORE_ACCEPTANCE_SCHEMA_ID",
    "CoreAcceptanceDecisionV1",
    "adjudicate_core_acceptance",
    "core_acceptance_from_json",
    "core_acceptance_to_json",
    "validate_core_acceptance_decision",
]
