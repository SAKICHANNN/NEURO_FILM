"""Fail-closed intake for a successor reference-match producer capability."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .contracts import ReferenceMatchContractError


SUCCESSOR_DECLARATION_SCHEMA = (
    "neuro-film.reference-match-successor-declaration.v1"
)
SUCCESSOR_POLICY_ID = "neuro-film.reference-match-successor-admission.v1"
REJECTED_CAPABILITY_ID = "zhuise.dpct-chroma.cpu-reference.v1"
REJECTED_WHEEL_SHA256 = (
    "fd995ad88c9f30f2136508f7c3879ce6768fde7ff2e149537d2b58f78e36c292"
)
REJECTED_STABLE_EVIDENCE_ID = (
    "90d0022c8c0d16070f38f96364e9f055ca86a3f5571a6eeafc780deea3f82d2a"
)
REJECTED_PRODUCER_STABLE_COMMIT = (
    "e22725d8524ed6ba56f37180abc400213908c6f4"
)
SUPPORTED_PROFILE_ID = "zhuise.display-linear-srgb-d65-relative-f32.v1"
SUPPORTED_LOWER_PROFILE_ID = "neuro-film.dpct-consumer.v2"
FROZEN_GATE_POLICY_ID = "neuro-film.reference-match-p44-gates.v1"
TARGET_RUNTIMES = (
    "windows_x64",
    "macos_arm64",
    "ios_arm64",
    "android_arm64",
)

_TOP_KEYS = {
    "schema_id",
    "policy_id",
    "candidate_id",
    "producer",
    "package",
    "wire",
    "semantics",
    "rights",
    "runtime_evidence",
    "product_evidence",
    "declaration_id",
}
_PRODUCER_KEYS = {
    "stable_commit",
    "package_source_commit",
    "fixture_commit",
    "package_lock_sha256",
    "conformance_fixture_sha256",
}
_PACKAGE_KEYS = {
    "distribution",
    "version",
    "entrypoint",
    "wheel_filename",
    "wheel_size_bytes",
    "wheel_sha256",
}
_WIRE_KEYS = {
    "capability_id",
    "profile_id",
    "lower_compatibility_profile_id",
    "request_schema",
    "request_schema_sha256",
    "response_schema",
    "response_schema_sha256",
}
_SEMANTIC_KEYS = {
    "fit_semantics",
    "batch_transform_policy",
    "deterministic",
    "hidden_state",
}
_RIGHTS_KEYS = {
    "evaluation_allowed",
    "commercial_use_allowed",
    "redistribution_allowed",
    "evidence_id",
}
_PRODUCT_EVIDENCE_KEYS = {
    "gate_policy_id",
    "stable_evidence_id",
    "a1_passed",
    "a4_passed",
    "a5_passed",
    "blind_aesthetic_passed",
}


@dataclass(frozen=True)
class SuccessorAdmissionDecisionV1:
    declaration_id: str
    evaluation_ready: bool
    product_ready: bool
    evaluation_reasons: tuple[str, ...]
    product_reasons: tuple[str, ...]


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "successor declaration is not canonicalizable"
        ) from exc


def _strict_mapping(
    value: Any,
    expected_keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the successor contract"
        )
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ReferenceMatchContractError(
            f"{label} must be a bounded non-empty string"
        )
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return text


def _commit(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise ReferenceMatchContractError(
            f"{label} must be a full lowercase Git commit"
        )
    return text


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ReferenceMatchContractError(f"{label} must be boolean")
    return value


def _validate_declaration(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    declaration = dict(_strict_mapping(value, _TOP_KEYS, "declaration"))
    if (
        declaration["schema_id"] != SUCCESSOR_DECLARATION_SCHEMA
        or declaration["policy_id"] != SUCCESSOR_POLICY_ID
    ):
        raise ReferenceMatchContractError(
            "successor declaration schema/policy is unsupported"
        )
    _nonempty_string(declaration["candidate_id"], "candidate_id")

    producer = _strict_mapping(
        declaration["producer"], _PRODUCER_KEYS, "producer"
    )
    for key in ("stable_commit", "package_source_commit", "fixture_commit"):
        _commit(producer[key], f"producer.{key}")
    for key in ("package_lock_sha256", "conformance_fixture_sha256"):
        _sha256(producer[key], f"producer.{key}")

    package = _strict_mapping(
        declaration["package"], _PACKAGE_KEYS, "package"
    )
    for key in (
        "distribution",
        "version",
        "entrypoint",
        "wheel_filename",
    ):
        _nonempty_string(package[key], f"package.{key}")
    if (
        isinstance(package["wheel_size_bytes"], bool)
        or not isinstance(package["wheel_size_bytes"], int)
        or not 1 <= package["wheel_size_bytes"] <= 1024 * 1024 * 1024
    ):
        raise ReferenceMatchContractError(
            "package.wheel_size_bytes is invalid"
        )
    _sha256(package["wheel_sha256"], "package.wheel_sha256")

    wire = _strict_mapping(declaration["wire"], _WIRE_KEYS, "wire")
    for key in (
        "capability_id",
        "profile_id",
        "lower_compatibility_profile_id",
        "request_schema",
        "response_schema",
    ):
        _nonempty_string(wire[key], f"wire.{key}")
    for key in ("request_schema_sha256", "response_schema_sha256"):
        _sha256(wire[key], f"wire.{key}")

    semantics = _strict_mapping(
        declaration["semantics"], _SEMANTIC_KEYS, "semantics"
    )
    fit_semantics = semantics["fit_semantics"]
    batch_policy = semantics["batch_transform_policy"]
    if fit_semantics not in {
        "source-reference-per-source",
        "reference-only-shared",
    }:
        raise ReferenceMatchContractError(
            "semantics.fit_semantics is unsupported"
        )
    expected_batch = {
        "source-reference-per-source": "per-source-bundle",
        "reference-only-shared": "shared-bundle",
    }[fit_semantics]
    if batch_policy != expected_batch:
        raise ReferenceMatchContractError(
            "fit and batch transform semantics are inconsistent"
        )
    if not _boolean(semantics["deterministic"], "semantics.deterministic"):
        raise ReferenceMatchContractError(
            "successor capability must declare deterministic execution"
        )
    if semantics["hidden_state"] != "none":
        raise ReferenceMatchContractError(
            "static-image successor capability must not carry hidden state"
        )

    rights = _strict_mapping(
        declaration["rights"], _RIGHTS_KEYS, "rights"
    )
    for key in (
        "evaluation_allowed",
        "commercial_use_allowed",
        "redistribution_allowed",
    ):
        _boolean(rights[key], f"rights.{key}")
    _nonempty_string(rights["evidence_id"], "rights.evidence_id")

    runtimes = _strict_mapping(
        declaration["runtime_evidence"],
        set(TARGET_RUNTIMES),
        "runtime_evidence",
    )
    for key in TARGET_RUNTIMES:
        _boolean(runtimes[key], f"runtime_evidence.{key}")

    product_evidence = declaration["product_evidence"]
    if product_evidence is not None:
        product_evidence = _strict_mapping(
            product_evidence,
            _PRODUCT_EVIDENCE_KEYS,
            "product_evidence",
        )
        _nonempty_string(
            product_evidence["gate_policy_id"],
            "product_evidence.gate_policy_id",
        )
        _sha256(
            product_evidence["stable_evidence_id"],
            "product_evidence.stable_evidence_id",
        )
        for key in (
            "a1_passed",
            "a4_passed",
            "a5_passed",
            "blind_aesthetic_passed",
        ):
            _boolean(product_evidence[key], f"product_evidence.{key}")

    claimed_id = _sha256(declaration["declaration_id"], "declaration_id")
    identity = dict(declaration)
    identity.pop("declaration_id")
    expected_id = hashlib.sha256(
        b"NeuroFilmReferenceMatchSuccessorDeclarationV1\0"
        + _canonical_json(identity)
    ).hexdigest()
    if claimed_id != expected_id:
        raise ReferenceMatchContractError(
            "successor declaration identity mismatch"
        )
    return declaration, claimed_id


def evaluate_successor_declaration_v1(
    value: Mapping[str, Any],
) -> SuccessorAdmissionDecisionV1:
    """Validate and classify a successor without invoking producer code."""

    declaration, declaration_id = _validate_declaration(value)
    producer = declaration["producer"]
    package = declaration["package"]
    wire = declaration["wire"]
    rights = declaration["rights"]
    runtimes = declaration["runtime_evidence"]
    evidence = declaration["product_evidence"]

    evaluation_reasons: list[str] = []
    if wire["capability_id"] == REJECTED_CAPABILITY_ID:
        evaluation_reasons.append("reuses-rejected-capability")
    if package["wheel_sha256"] == REJECTED_WHEEL_SHA256:
        evaluation_reasons.append("reuses-rejected-wheel")
    if producer["stable_commit"] == REJECTED_PRODUCER_STABLE_COMMIT:
        evaluation_reasons.append("reuses-rejected-producer-snapshot")
    if wire["profile_id"] != SUPPORTED_PROFILE_ID:
        evaluation_reasons.append("unsupported-colour-profile")
    if (
        wire["lower_compatibility_profile_id"]
        != SUPPORTED_LOWER_PROFILE_ID
    ):
        evaluation_reasons.append("unsupported-lower-compatibility-profile")
    if not rights["evaluation_allowed"]:
        evaluation_reasons.append("evaluation-rights-absent")

    product_reasons = list(evaluation_reasons)
    if evidence is None:
        product_reasons.append("product-evidence-absent")
    else:
        if evidence["gate_policy_id"] != FROZEN_GATE_POLICY_ID:
            product_reasons.append("gate-policy-mismatch")
        if evidence["stable_evidence_id"] == REJECTED_STABLE_EVIDENCE_ID:
            product_reasons.append("reuses-rejected-product-evidence")
        for gate in ("a1", "a4", "a5"):
            if not evidence[f"{gate}_passed"]:
                product_reasons.append(f"{gate}-not-passed")
        if not evidence["blind_aesthetic_passed"]:
            product_reasons.append("blind-aesthetic-review-not-passed")
    if not rights["commercial_use_allowed"]:
        product_reasons.append("commercial-rights-absent")
    if not rights["redistribution_allowed"]:
        product_reasons.append("redistribution-rights-absent")
    for runtime in TARGET_RUNTIMES:
        if not runtimes[runtime]:
            product_reasons.append(f"{runtime}-runtime-absent")

    return SuccessorAdmissionDecisionV1(
        declaration_id=declaration_id,
        evaluation_ready=not evaluation_reasons,
        product_ready=not product_reasons,
        evaluation_reasons=tuple(evaluation_reasons),
        product_reasons=tuple(product_reasons),
    )


def successor_declaration_id_v1(value: Mapping[str, Any]) -> str:
    """Build the canonical ID for a declaration lacking only its ID."""

    if "declaration_id" in value:
        raise ReferenceMatchContractError(
            "declaration_id must be omitted while computing identity"
        )
    return hashlib.sha256(
        b"NeuroFilmReferenceMatchSuccessorDeclarationV1\0"
        + _canonical_json(value)
    ).hexdigest()


__all__ = [
    "FROZEN_GATE_POLICY_ID",
    "REJECTED_CAPABILITY_ID",
    "REJECTED_PRODUCER_STABLE_COMMIT",
    "REJECTED_STABLE_EVIDENCE_ID",
    "REJECTED_WHEEL_SHA256",
    "SUCCESSOR_DECLARATION_SCHEMA",
    "SUCCESSOR_POLICY_ID",
    "SUPPORTED_LOWER_PROFILE_ID",
    "SUPPORTED_PROFILE_ID",
    "TARGET_RUNTIMES",
    "SuccessorAdmissionDecisionV1",
    "evaluate_successor_declaration_v1",
    "successor_declaration_id_v1",
]
