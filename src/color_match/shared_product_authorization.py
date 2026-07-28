"""Fail-closed product staging boundary for shared reference operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any, Mapping

from .strict_json import strict_json_loads
from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .promotion import PromotionDecision
from .shared_numeric_guard import (
    SharedNumericBatchGuardV1,
    validate_shared_numeric_batch_guard_v1,
)
from .shared_operator_batch import (
    SharedOperatorBatchV1,
    SharedReferenceOperatorV1,
    validate_shared_operator_batch_v1,
    validate_shared_reference_operator_v1,
)
from .core_contracts import MATCH_PROFILE_DISPLAY_SRGB
from .successor_admission import (
    FROZEN_GATE_POLICY_ID,
    SUPPORTED_PROFILE_ID,
    SuccessorAdmissionDecisionV1,
    evaluate_successor_declaration_v1,
)


SHARED_PROMOTION_BINDING_SCHEMA_ID = (
    "neuro-film.reference-shared-promotion-binding.v1"
)
SHARED_PRODUCT_AUTHORIZATION_SCHEMA_ID = (
    "neuro-film.reference-shared-product-staging-authorization.v1"
)
SHARED_PROMOTION_BINDING_CLAIM_CEILING = (
    "evidence-binding-only-no-authorization"
)
SHARED_PRODUCT_AUTHORIZATION_CLAIM_CEILING = (
    "staging-only-not-committed"
)
_HEX = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_BINDING_KEYS = {
    "schema_id",
    "binding_id",
    "declaration_id",
    "stable_evidence_id",
    "gate_policy_id",
    "capability_id",
    "producer_commit",
    "model_fingerprint",
    "options_sha256",
    "promotion_status",
    "promotion_reasons",
    "claim_ceiling",
}
_SOURCE_KEYS = {
    "source_index",
    "source_view_id",
    "receipt_id",
    "numeric_decision_id",
    "numeric_accepted_for_transaction",
    "action",
    "reasons",
}
_AUTHORIZATION_KEYS = {
    "schema_id",
    "authorization_id",
    "batch_id",
    "numeric_guard_batch_id",
    "operator_id",
    "declaration_id",
    "successor_declaration_json",
    "promotion_binding",
    "admission_evaluation_ready",
    "admission_product_ready",
    "admission_evaluation_reasons",
    "admission_product_reasons",
    "source_count",
    "state",
    "sources",
    "claim_ceiling",
}


@dataclass(frozen=True)
class SharedPromotionBindingV1:
    """Bind a consumer promotion decision to one shared model/options scope."""

    schema_id: str
    binding_id: str
    declaration_id: str
    stable_evidence_id: str
    gate_policy_id: str
    capability_id: str
    producer_commit: str
    model_fingerprint: str
    options_sha256: str
    promotion_status: str
    promotion_reasons: tuple[str, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["promotion_reasons"] = list(self.promotion_reasons)
        return payload


@dataclass(frozen=True)
class SharedProductAuthorizationSourceV1:
    source_index: int
    source_view_id: str
    receipt_id: str
    numeric_decision_id: str
    numeric_accepted_for_transaction: bool
    action: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class SharedProductStagingAuthorizationV1:
    """Authorize transaction staging only; never apply, commit or deliver."""

    schema_id: str
    authorization_id: str
    batch_id: str
    numeric_guard_batch_id: str
    operator_id: str
    declaration_id: str
    successor_declaration_json: str
    promotion_binding: SharedPromotionBindingV1
    admission_evaluation_ready: bool
    admission_product_ready: bool
    admission_evaluation_reasons: tuple[str, ...]
    admission_product_reasons: tuple[str, ...]
    source_count: int
    state: str
    sources: tuple[SharedProductAuthorizationSourceV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["promotion_binding"] = self.promotion_binding.to_dict()
        payload["admission_evaluation_reasons"] = list(
            self.admission_evaluation_reasons
        )
        payload["admission_product_reasons"] = list(
            self.admission_product_reasons
        )
        payload["sources"] = [source.to_dict() for source in self.sources]
        return payload


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the shared product contract"
        )
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ReferenceMatchContractError(
            f"{label} must be a bounded non-empty string"
        )
    return value


def _reasons(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or value != tuple(sorted(set(value)))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ReferenceMatchContractError(
            f"{label} must be sorted unique non-empty strings"
        )
    return value


def _identity_payload(value: Any, identity_key: str) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop(identity_key)
    return payload


def _canonical_declaration_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "successor declaration is not canonicalizable"
        ) from exc


def _declaration_product_evidence(
    declaration: Mapping[str, Any],
) -> Mapping[str, Any]:
    evidence = declaration.get("product_evidence")
    if not isinstance(evidence, Mapping):
        raise ReferenceMatchContractError(
            "shared promotion binding requires product evidence"
        )
    return evidence


def _validate_operator_declaration_binding(
    operator: SharedReferenceOperatorV1,
    declaration: Mapping[str, Any],
) -> SuccessorAdmissionDecisionV1:
    validate_shared_reference_operator_v1(operator)
    decision = evaluate_successor_declaration_v1(declaration)
    wire = declaration["wire"]
    producer = declaration["producer"]
    semantics = declaration["semantics"]
    if (
        semantics["fit_semantics"] != "reference-only-shared"
        or semantics["batch_transform_policy"] != "shared-bundle"
        or wire["capability_id"] != operator.capability_id
        or wire["profile_id"] != SUPPORTED_PROFILE_ID
        or operator.profile_id != MATCH_PROFILE_DISPLAY_SRGB
        or wire["lower_compatibility_profile_id"]
        != operator.compatibility_profile_id
        or producer["stable_commit"] != operator.producer_commit
    ):
        raise ReferenceMatchContractError(
            "successor declaration does not bind the shared operator"
        )
    return decision


def bind_shared_promotion_v1(
    *,
    operator: SharedReferenceOperatorV1,
    declaration: Mapping[str, Any],
    stable_evidence_id: str,
    promotion: PromotionDecision,
) -> SharedPromotionBindingV1:
    """Bind an independently adjudicated promotion to exact operator scope."""

    admission = _validate_operator_declaration_binding(
        operator, declaration
    )
    evidence = _declaration_product_evidence(declaration)
    stable_id = _hash(stable_evidence_id, "stable_evidence_id")
    if (
        admission.declaration_id != declaration["declaration_id"]
        or evidence["stable_evidence_id"] != stable_id
        or evidence["gate_policy_id"] != FROZEN_GATE_POLICY_ID
    ):
        raise ReferenceMatchContractError(
            "promotion evidence does not bind the successor declaration"
        )
    if not isinstance(promotion, PromotionDecision):
        raise ReferenceMatchContractError(
            "promotion must be a PromotionDecision"
        )
    if promotion.status not in {
        "rejected",
        "eligible-for-visual-review",
        "promoted",
    }:
        raise ReferenceMatchContractError(
            "shared promotion status is unsupported"
        )
    reasons = tuple(sorted(set(promotion.reasons)))
    if reasons != promotion.reasons:
        raise ReferenceMatchContractError(
            "shared promotion reasons must be sorted unique"
        )
    if (promotion.status == "promoted") != (not reasons):
        raise ReferenceMatchContractError(
            "shared promotion status/reasons are inconsistent"
        )
    provisional = SharedPromotionBindingV1(
        schema_id=SHARED_PROMOTION_BINDING_SCHEMA_ID,
        binding_id="0" * 64,
        declaration_id=admission.declaration_id,
        stable_evidence_id=stable_id,
        gate_policy_id=FROZEN_GATE_POLICY_ID,
        capability_id=operator.capability_id,
        producer_commit=operator.producer_commit,
        model_fingerprint=operator.model_fingerprint,
        options_sha256=operator.options_sha256,
        promotion_status=promotion.status,
        promotion_reasons=reasons,
        claim_ceiling=SHARED_PROMOTION_BINDING_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        binding_id=canonical_sha256(
            _identity_payload(provisional, "binding_id")
        ),
    )
    validate_shared_promotion_binding_v1(result)
    return result


def validate_shared_promotion_binding_v1(
    value: SharedPromotionBindingV1,
) -> None:
    if not isinstance(value, SharedPromotionBindingV1):
        raise ReferenceMatchContractError(
            "shared promotion binding type is invalid"
        )
    if value.schema_id != SHARED_PROMOTION_BINDING_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared promotion binding schema is invalid"
        )
    for field in (
        "binding_id",
        "declaration_id",
        "stable_evidence_id",
        "model_fingerprint",
        "options_sha256",
    ):
        _hash(getattr(value, field), field)
    if _COMMIT.fullmatch(value.producer_commit) is None:
        raise ReferenceMatchContractError(
            "shared promotion producer commit is invalid"
        )
    _text(value.gate_policy_id, "gate_policy_id")
    _text(value.capability_id, "capability_id")
    if value.gate_policy_id != FROZEN_GATE_POLICY_ID:
        raise ReferenceMatchContractError(
            "shared promotion gate policy is unsupported"
        )
    reasons = _reasons(value.promotion_reasons, "promotion_reasons")
    if value.promotion_status not in {
        "rejected",
        "eligible-for-visual-review",
        "promoted",
    }:
        raise ReferenceMatchContractError(
            "shared promotion status is unsupported"
        )
    if (value.promotion_status == "promoted") != (not reasons):
        raise ReferenceMatchContractError(
            "shared promotion status/reasons are inconsistent"
        )
    if (
        value.claim_ceiling
        != SHARED_PROMOTION_BINDING_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "shared promotion binding claim ceiling mismatch"
        )
    if value.binding_id != canonical_sha256(
        _identity_payload(value, "binding_id")
    ):
        raise ReferenceMatchContractError(
            "shared promotion binding identity mismatch"
        )


def _global_reasons(
    admission: SuccessorAdmissionDecisionV1,
    promotion: SharedPromotionBindingV1,
    numeric_guard: SharedNumericBatchGuardV1,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not admission.evaluation_ready:
        reasons.append("successor-evaluation-not-ready")
    if not admission.product_ready:
        reasons.append("successor-product-not-ready")
    if promotion.promotion_status != "promoted":
        reasons.append("promotion-not-promoted")
    if numeric_guard.atomic_state != "eligible-for-transaction":
        reasons.append("numeric-batch-fallback")
    return tuple(sorted(reasons))


def authorize_shared_product_staging_v1(
    *,
    batch: SharedOperatorBatchV1,
    numeric_guard: SharedNumericBatchGuardV1,
    declaration: Mapping[str, Any],
    promotion_binding: SharedPromotionBindingV1,
) -> SharedProductStagingAuthorizationV1:
    """Authorize a complete shared batch for staging or fall back atomically."""

    validate_shared_operator_batch_v1(batch)
    validate_shared_numeric_batch_guard_v1(numeric_guard)
    validate_shared_promotion_binding_v1(promotion_binding)
    admission = _validate_operator_declaration_binding(
        batch.operator, declaration
    )
    evidence = _declaration_product_evidence(declaration)
    if (
        numeric_guard.upstream_batch_id != batch.batch_id
        or numeric_guard.operator_id != batch.operator.operator_id
        or numeric_guard.source_count != batch.source_count
    ):
        raise ReferenceMatchContractError(
            "shared numeric guard does not bind the shared batch"
        )
    if (
        promotion_binding.declaration_id != admission.declaration_id
        or promotion_binding.stable_evidence_id
        != evidence["stable_evidence_id"]
        or promotion_binding.capability_id
        != batch.operator.capability_id
        or promotion_binding.producer_commit
        != batch.operator.producer_commit
        or promotion_binding.model_fingerprint
        != batch.operator.model_fingerprint
        or promotion_binding.options_sha256
        != batch.operator.options_sha256
    ):
        raise ReferenceMatchContractError(
            "promotion binding does not bind the shared batch"
        )
    global_reasons = _global_reasons(
        admission, promotion_binding, numeric_guard
    )
    rows: list[SharedProductAuthorizationSourceV1] = []
    for index, (receipt, decision) in enumerate(
        zip(batch.sources, numeric_guard.decisions, strict=True)
    ):
        if (
            receipt.source_index != index
            or decision.source_index != index
            or decision.source_view_id != receipt.source_view_id
            or decision.receipt_id != receipt.receipt_id
            or decision.operator_id != batch.operator.operator_id
        ):
            raise ReferenceMatchContractError(
                "shared authorization source binding mismatch"
            )
        reasons = list(global_reasons)
        reasons.extend(
            f"numeric:{reason}" for reason in decision.reasons
        )
        normalized = tuple(sorted(set(reasons)))
        rows.append(
            SharedProductAuthorizationSourceV1(
                source_index=index,
                source_view_id=receipt.source_view_id,
                receipt_id=receipt.receipt_id,
                numeric_decision_id=decision.decision_id,
                numeric_accepted_for_transaction=(
                    decision.accepted_for_transaction
                ),
                action=(
                    "authorized-for-staging"
                    if not global_reasons
                    else "identity-fallback"
                ),
                reasons=normalized,
            )
        )
    state = (
        "authorized-for-staging"
        if not global_reasons
        else "identity-fallback"
    )
    provisional = SharedProductStagingAuthorizationV1(
        schema_id=SHARED_PRODUCT_AUTHORIZATION_SCHEMA_ID,
        authorization_id="0" * 64,
        batch_id=batch.batch_id,
        numeric_guard_batch_id=numeric_guard.guard_batch_id,
        operator_id=batch.operator.operator_id,
        declaration_id=admission.declaration_id,
        successor_declaration_json=_canonical_declaration_json(declaration),
        promotion_binding=promotion_binding,
        admission_evaluation_ready=admission.evaluation_ready,
        admission_product_ready=admission.product_ready,
        admission_evaluation_reasons=tuple(
            sorted(set(admission.evaluation_reasons))
        ),
        admission_product_reasons=tuple(
            sorted(set(admission.product_reasons))
        ),
        source_count=batch.source_count,
        state=state,
        sources=tuple(rows),
        claim_ceiling=SHARED_PRODUCT_AUTHORIZATION_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        authorization_id=canonical_sha256(
            _identity_payload(provisional, "authorization_id")
        ),
    )
    validate_shared_product_staging_authorization_v1(result)
    return result


def validate_shared_product_staging_authorization_v1(
    value: SharedProductStagingAuthorizationV1,
) -> None:
    if not isinstance(value, SharedProductStagingAuthorizationV1):
        raise ReferenceMatchContractError(
            "shared product authorization type is invalid"
        )
    if value.schema_id != SHARED_PRODUCT_AUTHORIZATION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared product authorization schema is invalid"
        )
    for field in (
        "authorization_id",
        "batch_id",
        "numeric_guard_batch_id",
        "operator_id",
        "declaration_id",
    ):
        _hash(getattr(value, field), field)
    validate_shared_promotion_binding_v1(value.promotion_binding)
    if not isinstance(value.successor_declaration_json, str):
        raise ReferenceMatchContractError(
            "successor_declaration_json must be a string"
        )
    try:
        declaration = strict_json_loads(value.successor_declaration_json)
    except json.JSONDecodeError as exc:
        raise ReferenceMatchContractError(
            "successor declaration JSON is invalid"
        ) from exc
    if (
        not isinstance(declaration, Mapping)
        or _canonical_declaration_json(declaration)
        != value.successor_declaration_json
    ):
        raise ReferenceMatchContractError(
            "successor declaration JSON is not canonical"
        )
    admission = evaluate_successor_declaration_v1(declaration)
    evidence = _declaration_product_evidence(declaration)
    if (
        admission.declaration_id != value.declaration_id
        or value.promotion_binding.declaration_id != value.declaration_id
        or declaration["wire"]["capability_id"]
        != value.promotion_binding.capability_id
        or declaration["producer"]["stable_commit"]
        != value.promotion_binding.producer_commit
        or evidence["stable_evidence_id"]
        != value.promotion_binding.stable_evidence_id
    ):
        raise ReferenceMatchContractError(
            "shared authorization declaration binding mismatch"
        )
    if not isinstance(value.admission_evaluation_ready, bool) or not isinstance(
        value.admission_product_ready, bool
    ):
        raise ReferenceMatchContractError(
            "shared authorization admission states must be boolean"
        )
    evaluation_reasons = _reasons(
        value.admission_evaluation_reasons,
        "admission_evaluation_reasons",
    )
    product_reasons = _reasons(
        value.admission_product_reasons,
        "admission_product_reasons",
    )
    if (
        value.admission_evaluation_ready != admission.evaluation_ready
        or evaluation_reasons
        != tuple(sorted(set(admission.evaluation_reasons)))
    ):
        raise ReferenceMatchContractError(
            "shared authorization evaluation admission is inconsistent"
        )
    if (
        value.admission_product_ready != admission.product_ready
        or product_reasons != tuple(sorted(set(admission.product_reasons)))
    ):
        raise ReferenceMatchContractError(
            "shared authorization product admission is inconsistent"
        )
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.sources)
    ):
        raise ReferenceMatchContractError(
            "shared authorization source_count is invalid"
        )
    expected_global: list[str] = []
    if not value.admission_evaluation_ready:
        expected_global.append("successor-evaluation-not-ready")
    if not value.admission_product_ready:
        expected_global.append("successor-product-not-ready")
    if value.promotion_binding.promotion_status != "promoted":
        expected_global.append("promotion-not-promoted")
    any_numeric_rejected = any(
        not source.numeric_accepted_for_transaction
        for source in value.sources
    )
    if any_numeric_rejected:
        expected_global.append("numeric-batch-fallback")
    expected_global_tuple = tuple(sorted(expected_global))
    expected_state = (
        "authorized-for-staging"
        if not expected_global_tuple
        else "identity-fallback"
    )
    if value.state != expected_state:
        raise ReferenceMatchContractError(
            "shared authorization atomic state is inconsistent"
        )
    source_ids: set[str] = set()
    receipt_ids: set[str] = set()
    decision_ids: set[str] = set()
    for index, source in enumerate(value.sources):
        if not isinstance(source, SharedProductAuthorizationSourceV1):
            raise ReferenceMatchContractError(
                "shared authorization source type is invalid"
            )
        if source.source_index != index:
            raise ReferenceMatchContractError(
                "shared authorization source order is invalid"
            )
        for field in ("source_view_id", "receipt_id", "numeric_decision_id"):
            _hash(getattr(source, field), f"source.{field}")
        if (
            source.source_view_id in source_ids
            or source.receipt_id in receipt_ids
            or source.numeric_decision_id in decision_ids
        ):
            raise ReferenceMatchContractError(
                "shared authorization source identities must be distinct"
            )
        source_ids.add(source.source_view_id)
        receipt_ids.add(source.receipt_id)
        decision_ids.add(source.numeric_decision_id)
        if not isinstance(source.numeric_accepted_for_transaction, bool):
            raise ReferenceMatchContractError(
                "shared authorization numeric state must be boolean"
            )
        reasons = _reasons(source.reasons, "source.reasons")
        global_in_row = tuple(
            reason for reason in reasons if not reason.startswith("numeric:")
        )
        if global_in_row != expected_global_tuple:
            raise ReferenceMatchContractError(
                "shared authorization source global reasons mismatch"
            )
        if (
            source.numeric_accepted_for_transaction
            and any(reason.startswith("numeric:") for reason in reasons)
        ):
            raise ReferenceMatchContractError(
                "accepted numeric source cannot carry numeric reasons"
            )
        if (
            not source.numeric_accepted_for_transaction
            and not any(reason.startswith("numeric:") for reason in reasons)
        ):
            raise ReferenceMatchContractError(
                "rejected numeric source must carry numeric reasons"
            )
        expected_action = (
            "authorized-for-staging"
            if expected_state == "authorized-for-staging"
            else "identity-fallback"
        )
        if source.action != expected_action:
            raise ReferenceMatchContractError(
                "shared authorization source action is inconsistent"
            )
    if (
        value.claim_ceiling
        != SHARED_PRODUCT_AUTHORIZATION_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "shared authorization claim ceiling mismatch"
        )
    if value.authorization_id != canonical_sha256(
        _identity_payload(value, "authorization_id")
    ):
        raise ReferenceMatchContractError(
            "shared authorization identity mismatch"
        )


def shared_product_authorization_to_json(
    value: SharedProductStagingAuthorizationV1,
) -> str:
    validate_shared_product_staging_authorization_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def shared_product_authorization_from_json(
    encoded: str,
) -> SharedProductStagingAuthorizationV1:
    try:
        payload = strict_json_loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "shared product authorization is not valid JSON"
        ) from exc
    payload = _strict(payload, _AUTHORIZATION_KEYS, "authorization")
    raw_binding = dict(
        _strict(payload["promotion_binding"], _BINDING_KEYS, "binding")
    )
    raw_binding["promotion_reasons"] = tuple(
        raw_binding["promotion_reasons"]
    )
    try:
        binding = SharedPromotionBindingV1(**raw_binding)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "shared promotion binding fields are invalid"
        ) from exc
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ReferenceMatchContractError(
            "shared authorization sources must be a non-empty array"
        )
    sources: list[SharedProductAuthorizationSourceV1] = []
    for index, raw in enumerate(raw_sources):
        item = dict(_strict(raw, _SOURCE_KEYS, f"source {index}"))
        item["reasons"] = tuple(item["reasons"])
        try:
            sources.append(SharedProductAuthorizationSourceV1(**item))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "shared authorization source fields are invalid"
            ) from exc
    try:
        result = SharedProductStagingAuthorizationV1(
            schema_id=payload["schema_id"],
            authorization_id=payload["authorization_id"],
            batch_id=payload["batch_id"],
            numeric_guard_batch_id=payload["numeric_guard_batch_id"],
            operator_id=payload["operator_id"],
            declaration_id=payload["declaration_id"],
            successor_declaration_json=payload[
                "successor_declaration_json"
            ],
            promotion_binding=binding,
            admission_evaluation_ready=payload[
                "admission_evaluation_ready"
            ],
            admission_product_ready=payload["admission_product_ready"],
            admission_evaluation_reasons=tuple(
                payload["admission_evaluation_reasons"]
            ),
            admission_product_reasons=tuple(
                payload["admission_product_reasons"]
            ),
            source_count=payload["source_count"],
            state=payload["state"],
            sources=tuple(sources),
            claim_ceiling=payload["claim_ceiling"],
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "shared product authorization fields are invalid"
        ) from exc
    validate_shared_product_staging_authorization_v1(result)
    return result


__all__ = [
    "SHARED_PRODUCT_AUTHORIZATION_CLAIM_CEILING",
    "SHARED_PRODUCT_AUTHORIZATION_SCHEMA_ID",
    "SHARED_PROMOTION_BINDING_CLAIM_CEILING",
    "SHARED_PROMOTION_BINDING_SCHEMA_ID",
    "SharedProductAuthorizationSourceV1",
    "SharedProductStagingAuthorizationV1",
    "SharedPromotionBindingV1",
    "authorize_shared_product_staging_v1",
    "bind_shared_promotion_v1",
    "shared_product_authorization_from_json",
    "shared_product_authorization_to_json",
    "validate_shared_product_staging_authorization_v1",
    "validate_shared_promotion_binding_v1",
]
