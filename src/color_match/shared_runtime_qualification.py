"""Runtime-qualified guard over shared product staging authorization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from typing import Any, Mapping

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .shared_product_authorization import (
    SharedProductStagingAuthorizationV1,
    validate_shared_product_staging_authorization_v1,
)
from .successor_runtime_evidence import (
    SuccessorRuntimeEvidenceV1,
    bind_successor_runtime_evidence_v1,
    successor_runtime_evidence_from_json,
    successor_runtime_evidence_to_json,
    validate_successor_runtime_evidence_v1,
)


SHARED_RUNTIME_QUALIFICATION_SCHEMA_ID = (
    "neuro-film.reference-shared-runtime-qualification.v1"
)
SHARED_RUNTIME_QUALIFICATION_CLAIM_CEILING = (
    "runtime-qualified-staging-only-not-committed"
)
_STATES = {"runtime-qualified-for-staging", "identity-fallback"}
_KEYS = {
    "schema_id",
    "qualification_id",
    "upstream_authorization_id",
    "batch_id",
    "operator_id",
    "declaration_id",
    "successor_declaration_json",
    "runtime_evidence_id",
    "runtime_evidence_json",
    "runtime_ready",
    "runtime_satisfied_targets",
    "runtime_unsatisfied_targets",
    "runtime_reasons",
    "upstream_state",
    "state",
    "reasons",
    "claim_ceiling",
}


@dataclass(frozen=True)
class RuntimeQualifiedSharedAuthorizationV1:
    schema_id: str
    qualification_id: str
    upstream_authorization_id: str
    batch_id: str
    operator_id: str
    declaration_id: str
    successor_declaration_json: str
    runtime_evidence_id: str
    runtime_evidence_json: str
    runtime_ready: bool
    runtime_satisfied_targets: tuple[str, ...]
    runtime_unsatisfied_targets: tuple[str, ...]
    runtime_reasons: tuple[str, ...]
    upstream_state: str
    state: str
    reasons: tuple[str, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime_satisfied_targets"] = list(
            self.runtime_satisfied_targets
        )
        payload["runtime_unsatisfied_targets"] = list(
            self.runtime_unsatisfied_targets
        )
        payload["runtime_reasons"] = list(self.runtime_reasons)
        payload["reasons"] = list(self.reasons)
        return payload


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


def _parse_declaration(value: str) -> Mapping[str, Any]:
    if not isinstance(value, str):
        raise ReferenceMatchContractError(
            "successor declaration JSON must be a string"
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ReferenceMatchContractError(
            "successor declaration JSON is invalid"
        ) from exc
    if (
        not isinstance(parsed, Mapping)
        or _canonical_declaration_json(parsed) != value
    ):
        raise ReferenceMatchContractError(
            "successor declaration JSON is not canonical"
        )
    return parsed


def _ordered_texts(
    value: Any,
    label: str,
    *,
    sorted_unique: bool,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReferenceMatchContractError(f"{label} must be an array")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ReferenceMatchContractError(
            f"{label} contains an invalid string"
        )
    if sorted_unique and result != tuple(sorted(set(result))):
        raise ReferenceMatchContractError(
            f"{label} must be sorted and unique"
        )
    if not sorted_unique and len(result) != len(set(result)):
        raise ReferenceMatchContractError(
            f"{label} must not contain duplicates"
        )
    return result


def _identity_payload(
    value: RuntimeQualifiedSharedAuthorizationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("qualification_id")
    return payload


def _expected_reasons(
    upstream_state: str,
    runtime_reasons: tuple[str, ...],
) -> tuple[str, ...]:
    reasons = [f"runtime:{reason}" for reason in runtime_reasons]
    if upstream_state != "authorized-for-staging":
        reasons.append("upstream-authorization-not-ready")
    return tuple(sorted(set(reasons)))


def qualify_shared_product_authorization_runtime_v1(
    *,
    authorization: SharedProductStagingAuthorizationV1,
    runtime_evidence: SuccessorRuntimeEvidenceV1 | Mapping[str, Any],
) -> RuntimeQualifiedSharedAuthorizationV1:
    validate_shared_product_staging_authorization_v1(authorization)
    evidence = validate_successor_runtime_evidence_v1(runtime_evidence)
    declaration = _parse_declaration(
        authorization.successor_declaration_json
    )
    runtime = bind_successor_runtime_evidence_v1(declaration, evidence)
    reasons = _expected_reasons(authorization.state, runtime.reasons)
    state = (
        "runtime-qualified-for-staging"
        if not reasons
        else "identity-fallback"
    )
    provisional = RuntimeQualifiedSharedAuthorizationV1(
        schema_id=SHARED_RUNTIME_QUALIFICATION_SCHEMA_ID,
        qualification_id="0" * 64,
        upstream_authorization_id=authorization.authorization_id,
        batch_id=authorization.batch_id,
        operator_id=authorization.operator_id,
        declaration_id=authorization.declaration_id,
        successor_declaration_json=authorization.successor_declaration_json,
        runtime_evidence_id=evidence.evidence_id,
        runtime_evidence_json=successor_runtime_evidence_to_json(evidence),
        runtime_ready=runtime.runtime_ready,
        runtime_satisfied_targets=runtime.satisfied_targets,
        runtime_unsatisfied_targets=runtime.unsatisfied_targets,
        runtime_reasons=tuple(sorted(runtime.reasons)),
        upstream_state=authorization.state,
        state=state,
        reasons=reasons,
        claim_ceiling=SHARED_RUNTIME_QUALIFICATION_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        qualification_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_runtime_qualified_shared_authorization_binding_v1(
        result,
        authorization=authorization,
    )
    return result


def validate_runtime_qualified_shared_authorization_v1(
    value: RuntimeQualifiedSharedAuthorizationV1,
) -> None:
    if not isinstance(value, RuntimeQualifiedSharedAuthorizationV1):
        raise ReferenceMatchContractError(
            "runtime qualification type is invalid"
        )
    if value.schema_id != SHARED_RUNTIME_QUALIFICATION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "runtime qualification schema is invalid"
        )
    for field in (
        "qualification_id",
        "upstream_authorization_id",
        "batch_id",
        "operator_id",
        "declaration_id",
        "runtime_evidence_id",
    ):
        _hash(getattr(value, field), field)
    declaration = _parse_declaration(value.successor_declaration_json)
    evidence = successor_runtime_evidence_from_json(
        value.runtime_evidence_json
    )
    if successor_runtime_evidence_to_json(evidence) != value.runtime_evidence_json:
        raise ReferenceMatchContractError(
            "runtime evidence JSON is not canonical"
        )
    runtime = bind_successor_runtime_evidence_v1(declaration, evidence)
    if (
        declaration["declaration_id"] != value.declaration_id
        or evidence.evidence_id != value.runtime_evidence_id
        or runtime.declaration_id != value.declaration_id
    ):
        raise ReferenceMatchContractError(
            "runtime qualification evidence binding mismatch"
        )
    if not isinstance(value.runtime_ready, bool):
        raise ReferenceMatchContractError(
            "runtime qualification readiness must be boolean"
        )
    satisfied = _ordered_texts(
        value.runtime_satisfied_targets,
        "runtime_satisfied_targets",
        sorted_unique=False,
    )
    unsatisfied = _ordered_texts(
        value.runtime_unsatisfied_targets,
        "runtime_unsatisfied_targets",
        sorted_unique=False,
    )
    runtime_reasons = _ordered_texts(
        value.runtime_reasons,
        "runtime_reasons",
        sorted_unique=True,
    )
    reasons = _ordered_texts(
        value.reasons,
        "reasons",
        sorted_unique=True,
    )
    if (
        value.runtime_ready != runtime.runtime_ready
        or satisfied != runtime.satisfied_targets
        or unsatisfied != runtime.unsatisfied_targets
        or runtime_reasons != tuple(sorted(runtime.reasons))
    ):
        raise ReferenceMatchContractError(
            "runtime qualification decision facts are inconsistent"
        )
    if value.upstream_state not in {
        "authorized-for-staging",
        "identity-fallback",
    }:
        raise ReferenceMatchContractError(
            "runtime qualification upstream state is invalid"
        )
    expected_reasons = _expected_reasons(
        value.upstream_state,
        runtime_reasons,
    )
    expected_state = (
        "runtime-qualified-for-staging"
        if not expected_reasons
        else "identity-fallback"
    )
    if reasons != expected_reasons or value.state != expected_state:
        raise ReferenceMatchContractError(
            "runtime qualification state/reasons are inconsistent"
        )
    if value.state not in _STATES:
        raise ReferenceMatchContractError(
            "runtime qualification state is invalid"
        )
    if (
        value.claim_ceiling
        != SHARED_RUNTIME_QUALIFICATION_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "runtime qualification claim ceiling is invalid"
        )
    if value.qualification_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "runtime qualification identity mismatch"
        )


def validate_runtime_qualified_shared_authorization_binding_v1(
    value: RuntimeQualifiedSharedAuthorizationV1,
    *,
    authorization: SharedProductStagingAuthorizationV1,
) -> None:
    validate_runtime_qualified_shared_authorization_v1(value)
    validate_shared_product_staging_authorization_v1(authorization)
    if (
        value.upstream_authorization_id != authorization.authorization_id
        or value.batch_id != authorization.batch_id
        or value.operator_id != authorization.operator_id
        or value.declaration_id != authorization.declaration_id
        or value.successor_declaration_json
        != authorization.successor_declaration_json
        or value.upstream_state != authorization.state
    ):
        raise ReferenceMatchContractError(
            "runtime qualification does not bind the upstream authorization"
        )


def runtime_qualified_shared_authorization_to_dict(
    value: RuntimeQualifiedSharedAuthorizationV1,
) -> dict[str, Any]:
    validate_runtime_qualified_shared_authorization_v1(value)
    return value.to_dict()


def runtime_qualified_shared_authorization_to_json(
    value: RuntimeQualifiedSharedAuthorizationV1,
) -> str:
    return json.dumps(
        runtime_qualified_shared_authorization_to_dict(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def runtime_qualified_shared_authorization_from_dict(
    value: Mapping[str, Any],
) -> RuntimeQualifiedSharedAuthorizationV1:
    if not isinstance(value, Mapping) or set(value) != _KEYS:
        raise ReferenceMatchContractError(
            "runtime qualification fields differ from the contract"
        )
    arrays: dict[str, tuple[Any, ...]] = {}
    for field in (
        "runtime_satisfied_targets",
        "runtime_unsatisfied_targets",
        "runtime_reasons",
        "reasons",
    ):
        raw = value[field]
        if not isinstance(raw, (list, tuple)):
            raise ReferenceMatchContractError(
                f"{field} must be an array"
            )
        arrays[field] = tuple(raw)
    result = RuntimeQualifiedSharedAuthorizationV1(
        schema_id=value["schema_id"],
        qualification_id=value["qualification_id"],
        upstream_authorization_id=value["upstream_authorization_id"],
        batch_id=value["batch_id"],
        operator_id=value["operator_id"],
        declaration_id=value["declaration_id"],
        successor_declaration_json=value["successor_declaration_json"],
        runtime_evidence_id=value["runtime_evidence_id"],
        runtime_evidence_json=value["runtime_evidence_json"],
        runtime_ready=value["runtime_ready"],
        runtime_satisfied_targets=arrays["runtime_satisfied_targets"],
        runtime_unsatisfied_targets=arrays["runtime_unsatisfied_targets"],
        runtime_reasons=arrays["runtime_reasons"],
        upstream_state=value["upstream_state"],
        state=value["state"],
        reasons=arrays["reasons"],
        claim_ceiling=value["claim_ceiling"],
    )
    validate_runtime_qualified_shared_authorization_v1(result)
    return result


def runtime_qualified_shared_authorization_from_json(
    value: str,
) -> RuntimeQualifiedSharedAuthorizationV1:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "runtime qualification JSON is invalid"
        ) from exc
    return runtime_qualified_shared_authorization_from_dict(parsed)


__all__ = [
    "SHARED_RUNTIME_QUALIFICATION_CLAIM_CEILING",
    "SHARED_RUNTIME_QUALIFICATION_SCHEMA_ID",
    "RuntimeQualifiedSharedAuthorizationV1",
    "qualify_shared_product_authorization_runtime_v1",
    "runtime_qualified_shared_authorization_from_dict",
    "runtime_qualified_shared_authorization_from_json",
    "runtime_qualified_shared_authorization_to_dict",
    "runtime_qualified_shared_authorization_to_json",
    "validate_runtime_qualified_shared_authorization_binding_v1",
    "validate_runtime_qualified_shared_authorization_v1",
]
