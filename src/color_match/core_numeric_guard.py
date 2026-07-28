"""Numeric delivered-pixel guard for exact external-core receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from typing import Any, Mapping

import numpy as np

from .strict_json import strict_json_loads
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
)


CORE_NUMERIC_GUARD_POLICY_SCHEMA_ID = (
    "neuro-film.core-numeric-guard-policy.v1"
)
CORE_NUMERIC_GUARD_DECISION_SCHEMA_ID = (
    "neuro-film.core-numeric-guard-decision.v1"
)
CORE_NUMERIC_GUARD_CLAIM_CEILING = "numeric-only-no-visual-claim"
_DECISION_KEYS = {
    "schema_id",
    "decision_id",
    "policy",
    "policy_id",
    "admission_id",
    "receipt_id",
    "source_view_id",
    "output_view_id",
    "admission_accepted_for_product_guard",
    "accepted_for_transaction",
    "action",
    "reasons",
    "out_of_gamut_fraction",
    "clipping_fraction",
    "projected_fraction",
    "new_boundary_fraction",
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
class CoreNumericGuardPolicyV1:
    schema_id: str = CORE_NUMERIC_GUARD_POLICY_SCHEMA_ID
    max_out_of_gamut_fraction: float = 0.25
    max_clipping_fraction: float = 0.05
    max_projected_fraction: float = 0.25
    max_new_boundary_fraction: float = 0.05
    boundary_epsilon: float = 1.0 / 65535.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoreNumericGuardDecisionV1:
    """Exact receipt guard result; success is not final delivery."""

    schema_id: str
    decision_id: str
    policy: CoreNumericGuardPolicyV1
    policy_id: str
    admission_id: str
    receipt_id: str
    source_view_id: str
    output_view_id: str
    admission_accepted_for_product_guard: bool
    accepted_for_transaction: bool
    action: str
    reasons: tuple[str, ...]
    out_of_gamut_fraction: float
    clipping_fraction: float
    projected_fraction: float
    new_boundary_fraction: float
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
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


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceMatchContractError(
            f"{label} must be finite within [0, 1]"
        )
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ReferenceMatchContractError(
            f"{label} must be finite within [0, 1]"
        )
    return result


def validate_core_numeric_guard_policy_v1(
    value: CoreNumericGuardPolicyV1,
) -> None:
    if not isinstance(value, CoreNumericGuardPolicyV1):
        raise ReferenceMatchContractError(
            "numeric guard policy must be CoreNumericGuardPolicyV1"
        )
    if value.schema_id != CORE_NUMERIC_GUARD_POLICY_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported numeric guard policy schema"
        )
    for label in (
        "max_out_of_gamut_fraction",
        "max_clipping_fraction",
        "max_projected_fraction",
        "max_new_boundary_fraction",
    ):
        _fraction(getattr(value, label), f"policy.{label}")
    epsilon = value.boundary_epsilon
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not np.isfinite(float(epsilon))
        or float(epsilon) < 0.0
        or float(epsilon) >= 0.5
    ):
        raise ReferenceMatchContractError(
            "policy.boundary_epsilon must be finite within [0, 0.5)"
        )


def core_numeric_guard_policy_id(
    value: CoreNumericGuardPolicyV1,
) -> str:
    validate_core_numeric_guard_policy_v1(value)
    return canonical_sha256(value.to_dict())


def _new_boundary_fraction(
    source_pixels: np.ndarray,
    candidate_pixels: np.ndarray,
    *,
    epsilon: float,
) -> float:
    source_boundary = np.any(
        (source_pixels <= epsilon)
        | (source_pixels >= 1.0 - epsilon),
        axis=-1,
    )
    candidate_boundary = np.any(
        (candidate_pixels <= epsilon)
        | (candidate_pixels >= 1.0 - epsilon),
        axis=-1,
    )
    return float(
        np.mean(candidate_boundary & ~source_boundary, dtype=np.float64)
    )


def _identity_payload(
    value: CoreNumericGuardDecisionV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("decision_id")
    return payload


def guard_core_candidate_numeric_v1(
    *,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    candidate: AdaptedDpctCandidateV2,
    acceptance: CoreAcceptanceDecisionV1,
    admission: CoreCandidateAdmissionV2,
    policy: CoreNumericGuardPolicyV1 | None = None,
) -> CoreNumericGuardDecisionV1:
    """Evaluate exact pixels without claiming visual or aesthetic safety."""

    resolved = policy or CoreNumericGuardPolicyV1()
    validate_core_numeric_guard_policy_v1(resolved)
    validate_prepared_match_view(source)
    validate_prepared_match_view(reference)
    if not isinstance(candidate, AdaptedDpctCandidateV2):
        raise ReferenceMatchContractError(
            "numeric guard candidate type is invalid"
        )
    if (
        candidate.aliases.compatibility_profile_id
        != DPCT_COMPATIBILITY_PROFILE_ID
        or candidate.aliases.producer_commit != DPCT_PINNED_COMMIT
    ):
        raise ReferenceMatchContractError(
            "numeric guard candidate compatibility identity mismatch"
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
    validate_core_acceptance_decision(acceptance)
    validate_core_candidate_admission_binding(
        admission,
        receipt=candidate.prepared_output.receipt,
        acceptance=acceptance,
    )
    new_boundary = _new_boundary_fraction(
        source.pixels,
        candidate.prepared_output.pixels,
        epsilon=float(resolved.boundary_epsilon),
    )
    diagnostics = candidate.diagnostics
    reasons: list[str] = []
    if not admission.accepted_for_product_guard:
        reasons.append("admission-identity-fallback")
    if (
        diagnostics.out_of_gamut_fraction
        > resolved.max_out_of_gamut_fraction
    ):
        reasons.append("out-of-gamut-fraction")
    if diagnostics.clipping_fraction > resolved.max_clipping_fraction:
        reasons.append("clipping-fraction")
    if diagnostics.projected_fraction > resolved.max_projected_fraction:
        reasons.append("projected-fraction")
    if new_boundary > resolved.max_new_boundary_fraction:
        reasons.append("new-boundary-fraction")
    normalized_reasons = tuple(sorted(set(reasons)))
    accepted = not normalized_reasons
    provisional = CoreNumericGuardDecisionV1(
        schema_id=CORE_NUMERIC_GUARD_DECISION_SCHEMA_ID,
        decision_id="0" * 64,
        policy=resolved,
        policy_id=core_numeric_guard_policy_id(resolved),
        admission_id=admission.admission_id,
        receipt_id=candidate.prepared_output.receipt.receipt_id,
        source_view_id=source.descriptor.view_id,
        output_view_id=(
            candidate.prepared_output.receipt.output_view.view_id
        ),
        admission_accepted_for_product_guard=(
            admission.accepted_for_product_guard
        ),
        accepted_for_transaction=accepted,
        action=(
            "eligible-for-transaction"
            if accepted
            else "identity-fallback"
        ),
        reasons=normalized_reasons,
        out_of_gamut_fraction=diagnostics.out_of_gamut_fraction,
        clipping_fraction=diagnostics.clipping_fraction,
        projected_fraction=diagnostics.projected_fraction,
        new_boundary_fraction=new_boundary,
        claim_ceiling=CORE_NUMERIC_GUARD_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        decision_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_core_numeric_guard_decision_v1(result)
    return result


def validate_core_numeric_guard_decision_v1(
    value: CoreNumericGuardDecisionV1,
) -> None:
    if not isinstance(value, CoreNumericGuardDecisionV1):
        raise ReferenceMatchContractError(
            "numeric guard decision must be CoreNumericGuardDecisionV1"
        )
    if value.schema_id != CORE_NUMERIC_GUARD_DECISION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported numeric guard decision schema"
        )
    _sha256(value.decision_id, "decision_id")
    validate_core_numeric_guard_policy_v1(value.policy)
    if value.policy_id != core_numeric_guard_policy_id(value.policy):
        raise ReferenceMatchContractError(
            "numeric guard policy_id mismatch"
        )
    for label in (
        "admission_id",
        "receipt_id",
        "source_view_id",
        "output_view_id",
    ):
        _sha256(getattr(value, label), label)
    if not isinstance(value.admission_accepted_for_product_guard, bool):
        raise ReferenceMatchContractError(
            "admission_accepted_for_product_guard must be boolean"
        )
    if not isinstance(value.accepted_for_transaction, bool):
        raise ReferenceMatchContractError(
            "accepted_for_transaction must be boolean"
        )
    if value.action not in {
        "eligible-for-transaction",
        "identity-fallback",
    }:
        raise ReferenceMatchContractError(
            "numeric guard action is unsupported"
        )
    if (
        not isinstance(value.reasons, tuple)
        or any(not isinstance(item, str) or not item for item in value.reasons)
        or value.reasons != tuple(sorted(set(value.reasons)))
    ):
        raise ReferenceMatchContractError(
            "numeric guard reasons must be sorted unique strings"
        )
    for label in (
        "out_of_gamut_fraction",
        "clipping_fraction",
        "projected_fraction",
        "new_boundary_fraction",
    ):
        _fraction(getattr(value, label), label)
    if value.claim_ceiling != CORE_NUMERIC_GUARD_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "numeric guard claim ceiling mismatch"
        )
    expected_accepted = not value.reasons
    expected_action = (
        "eligible-for-transaction"
        if expected_accepted
        else "identity-fallback"
    )
    if (
        value.accepted_for_transaction != expected_accepted
        or value.action != expected_action
    ):
        raise ReferenceMatchContractError(
            "numeric guard decision state is inconsistent"
        )
    policy = value.policy
    expected_reasons: list[str] = []
    if not value.admission_accepted_for_product_guard:
        expected_reasons.append("admission-identity-fallback")
    if (
        value.out_of_gamut_fraction
        > policy.max_out_of_gamut_fraction
    ):
        expected_reasons.append("out-of-gamut-fraction")
    if value.clipping_fraction > policy.max_clipping_fraction:
        expected_reasons.append("clipping-fraction")
    if value.projected_fraction > policy.max_projected_fraction:
        expected_reasons.append("projected-fraction")
    if value.new_boundary_fraction > policy.max_new_boundary_fraction:
        expected_reasons.append("new-boundary-fraction")
    if value.reasons != tuple(sorted(expected_reasons)):
        raise ReferenceMatchContractError(
            "numeric guard reasons are inconsistent"
        )
    if value.decision_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "numeric guard decision_id mismatch"
        )


def core_numeric_guard_decision_to_json(
    value: CoreNumericGuardDecisionV1,
) -> str:
    validate_core_numeric_guard_decision_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def core_numeric_guard_decision_from_json(
    encoded: str,
) -> CoreNumericGuardDecisionV1:
    try:
        payload = strict_json_loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "numeric guard decision is not valid JSON"
        ) from exc
    payload = _strict(payload, _DECISION_KEYS, "numeric guard decision")
    raw_policy = _strict(
        payload["policy"], _POLICY_KEYS, "numeric guard policy"
    )
    try:
        policy = CoreNumericGuardPolicyV1(**dict(raw_policy))
        converted = dict(payload)
        converted["policy"] = policy
        converted["reasons"] = tuple(converted["reasons"])
        value = CoreNumericGuardDecisionV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "numeric guard decision fields are invalid"
        ) from exc
    validate_core_numeric_guard_decision_v1(value)
    return value


__all__ = [
    "CORE_NUMERIC_GUARD_CLAIM_CEILING",
    "CORE_NUMERIC_GUARD_DECISION_SCHEMA_ID",
    "CORE_NUMERIC_GUARD_POLICY_SCHEMA_ID",
    "CoreNumericGuardDecisionV1",
    "CoreNumericGuardPolicyV1",
    "core_numeric_guard_decision_from_json",
    "core_numeric_guard_decision_to_json",
    "core_numeric_guard_policy_id",
    "guard_core_candidate_numeric_v1",
    "validate_core_numeric_guard_decision_v1",
    "validate_core_numeric_guard_policy_v1",
]
