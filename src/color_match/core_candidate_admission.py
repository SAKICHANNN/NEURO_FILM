"""Receipt-bound admission into the existing delivered-pixel guard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from typing import Any, Mapping

from .strict_json import strict_json_loads
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_acceptance import (
    CoreAcceptanceDecisionV1,
    validate_core_acceptance_decision,
)
from .core_apply_receipt import (
    CoreApplyReceiptV1,
    PreparedCoreApplyReceiptV1,
    validate_core_apply_receipt,
    validate_core_apply_receipt_binding,
    validate_prepared_core_apply_receipt,
)
from .core_contracts import (
    CapabilitiesV1,
    DiagnosticsV1,
    MatchViewV1,
    TransformBundleV1,
)


CORE_CANDIDATE_ADMISSION_SCHEMA_ID = (
    "neuro-film.reference-core-candidate-admission.v2"
)
CORE_CANDIDATE_ADMISSION_POLICY_ID = (
    "reference-core-candidate-admission.v2"
)
_REQUIRED_GATES = ("A1", "A4", "A5")
_KEYS = {
    "schema_id",
    "admission_id",
    "policy_id",
    "apply_receipt_id",
    "acceptance_decision_id",
    "transform_id",
    "output_view_id",
    "accepted_for_product_guard",
    "action",
    "guard_state",
    "required_gates",
}


@dataclass(frozen=True)
class CoreCandidateAdmissionV2:
    """Exact-pixel candidate admission; never a delivered result."""

    schema_id: str
    admission_id: str
    policy_id: str
    apply_receipt_id: str
    acceptance_decision_id: str
    transform_id: str
    output_view_id: str
    accepted_for_product_guard: bool
    action: str
    guard_state: str
    required_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_gates"] = list(self.required_gates)
        return payload


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


def _identity_payload(
    value: CoreCandidateAdmissionV2,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("admission_id")
    return payload


def admit_core_apply_receipt(
    *,
    prepared: PreparedCoreApplyReceiptV1,
    acceptance: CoreAcceptanceDecisionV1,
    source: MatchViewV1,
    reference: MatchViewV1,
    transform: TransformBundleV1,
    capabilities: CapabilitiesV1,
    diagnostics: DiagnosticsV1,
) -> CoreCandidateAdmissionV2:
    """Bind exact prepared pixels to one prior A1/A4/A5 decision."""

    validate_prepared_core_apply_receipt(prepared)
    validate_core_apply_receipt_binding(
        prepared.receipt,
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
    )
    validate_core_acceptance_decision(acceptance)
    if acceptance.transform_id != prepared.receipt.transform_id:
        raise ReferenceMatchContractError(
            "acceptance transform does not match apply receipt"
        )
    accepted = acceptance.accepted_for_product_guard
    provisional = CoreCandidateAdmissionV2(
        schema_id=CORE_CANDIDATE_ADMISSION_SCHEMA_ID,
        admission_id="0" * 64,
        policy_id=CORE_CANDIDATE_ADMISSION_POLICY_ID,
        apply_receipt_id=prepared.receipt.receipt_id,
        acceptance_decision_id=acceptance.decision_id,
        transform_id=transform.transform_id,
        output_view_id=prepared.receipt.output_view.view_id,
        accepted_for_product_guard=accepted,
        action=acceptance.action,
        guard_state=(
            "pending-product-guard"
            if accepted
            else "identity-fallback"
        ),
        required_gates=_REQUIRED_GATES,
    )
    result = replace(
        provisional,
        admission_id=canonical_sha256(
            _identity_payload(provisional)
        ),
    )
    validate_core_candidate_admission(result)
    validate_core_candidate_admission_binding(
        result,
        receipt=prepared.receipt,
        acceptance=acceptance,
    )
    return result


def validate_core_candidate_admission(
    value: CoreCandidateAdmissionV2,
) -> None:
    if not isinstance(value, CoreCandidateAdmissionV2):
        raise ReferenceMatchContractError(
            "core candidate admission must be CoreCandidateAdmissionV2"
        )
    if value.schema_id != CORE_CANDIDATE_ADMISSION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported core candidate admission schema"
        )
    _sha256(value.admission_id, "admission_id")
    if value.policy_id != CORE_CANDIDATE_ADMISSION_POLICY_ID:
        raise ReferenceMatchContractError(
            "unsupported core candidate admission policy"
        )
    _sha256(value.apply_receipt_id, "apply_receipt_id")
    _sha256(value.acceptance_decision_id, "acceptance_decision_id")
    _sha256(value.transform_id, "transform_id")
    _sha256(value.output_view_id, "output_view_id")
    if not isinstance(value.accepted_for_product_guard, bool):
        raise ReferenceMatchContractError(
            "accepted_for_product_guard must be boolean"
        )
    if value.action not in {
        "candidate-for-product-guard",
        "identity-fallback",
    }:
        raise ReferenceMatchContractError(
            "unsupported core candidate admission action"
        )
    if value.guard_state not in {
        "pending-product-guard",
        "identity-fallback",
    }:
        raise ReferenceMatchContractError(
            "unsupported core candidate guard state"
        )
    if value.required_gates != _REQUIRED_GATES:
        raise ReferenceMatchContractError(
            "core candidate admission must bind A1/A4/A5"
        )
    expected_action = (
        "candidate-for-product-guard"
        if value.accepted_for_product_guard
        else "identity-fallback"
    )
    expected_guard = (
        "pending-product-guard"
        if value.accepted_for_product_guard
        else "identity-fallback"
    )
    if (
        value.action != expected_action
        or value.guard_state != expected_guard
    ):
        raise ReferenceMatchContractError(
            "core candidate admission state is inconsistent"
        )
    if value.admission_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "admission_id does not match canonical payload"
        )


def validate_core_candidate_admission_binding(
    value: CoreCandidateAdmissionV2,
    *,
    receipt: CoreApplyReceiptV1,
    acceptance: CoreAcceptanceDecisionV1,
) -> None:
    validate_core_candidate_admission(value)
    validate_core_apply_receipt(receipt)
    validate_core_acceptance_decision(acceptance)
    identities = (
        value.apply_receipt_id == receipt.receipt_id,
        value.acceptance_decision_id == acceptance.decision_id,
        value.transform_id == receipt.transform_id,
        value.transform_id == acceptance.transform_id,
        value.output_view_id == receipt.output_view.view_id,
        value.accepted_for_product_guard
        == acceptance.accepted_for_product_guard,
        value.action == acceptance.action,
    )
    if not all(identities):
        raise ReferenceMatchContractError(
            "core candidate admission binding mismatch"
        )


def core_candidate_admission_to_json(
    value: CoreCandidateAdmissionV2,
) -> str:
    validate_core_candidate_admission(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def core_candidate_admission_from_json(
    encoded: str,
) -> CoreCandidateAdmissionV2:
    try:
        payload = strict_json_loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "core candidate admission is not valid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReferenceMatchContractError(
            "core candidate admission must be an object"
        )
    _strict_keys(payload, _KEYS, "core candidate admission")
    converted = dict(payload)
    converted["required_gates"] = tuple(converted["required_gates"])
    try:
        result = CoreCandidateAdmissionV2(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "core candidate admission contains invalid fields"
        ) from exc
    validate_core_candidate_admission(result)
    return result


__all__ = [
    "CORE_CANDIDATE_ADMISSION_POLICY_ID",
    "CORE_CANDIDATE_ADMISSION_SCHEMA_ID",
    "CoreCandidateAdmissionV2",
    "admit_core_apply_receipt",
    "core_candidate_admission_from_json",
    "core_candidate_admission_to_json",
    "validate_core_candidate_admission",
    "validate_core_candidate_admission_binding",
]
