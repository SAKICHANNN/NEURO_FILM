"""Source-bound target-runtime evidence for successor producer capabilities.

This module validates evidence identities only. It does not execute, compile,
load or otherwise implement a producer-native backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .contracts import ReferenceMatchContractError
from .successor_admission import (
    TARGET_RUNTIMES,
    evaluate_successor_declaration_v1,
)


RUNTIME_EVIDENCE_SCHEMA_ID = (
    "neuro-film.reference-match-successor-runtime-evidence.v1"
)
RUNTIME_EVIDENCE_POLICY_ID = (
    "neuro-film.reference-match-successor-runtime-evidence-policy.v1"
)
RUNTIME_EVIDENCE_CLAIM_CEILING = (
    "runtime-evidence-only-no-algorithm-promotion"
)
RUNTIME_PROOF_CLASSES = (
    "host-runtime",
    "device-runtime",
    "cross-compiled",
    "link-only",
    "object-only",
)
REQUIRED_RUNTIME_PROOF_CLASS = {
    "windows_x64": "host-runtime",
    "macos_arm64": "host-runtime",
    "ios_arm64": "device-runtime",
    "android_arm64": "device-runtime",
}

_BUNDLE_KEYS = {
    "schema_id",
    "policy_id",
    "declaration_id",
    "producer_commit",
    "capability_id",
    "profile_id",
    "records",
    "evidence_id",
    "claim_ceiling",
}
_RECORD_KEYS = {
    "target_runtime",
    "proof_class",
    "os_name",
    "os_version",
    "architecture",
    "environment_count",
    "environment_matrix_sha256",
    "environment_summary",
    "backend_id",
    "backend_version",
    "runner_sha256",
    "executable_sha256",
    "report_sha256",
    "replay_count",
    "deterministic_replay_passed",
    "conformance_passed",
    "failure_injection_passed",
    "record_id",
}


@dataclass(frozen=True)
class SuccessorRuntimeEvidenceRecordV1:
    target_runtime: str
    proof_class: str
    os_name: str
    os_version: str
    architecture: str
    environment_count: int
    environment_matrix_sha256: str
    environment_summary: str
    backend_id: str
    backend_version: str
    runner_sha256: str
    executable_sha256: str
    report_sha256: str
    replay_count: int
    deterministic_replay_passed: bool
    conformance_passed: bool
    failure_injection_passed: bool
    record_id: str


@dataclass(frozen=True)
class SuccessorRuntimeEvidenceV1:
    schema_id: str
    policy_id: str
    declaration_id: str
    producer_commit: str
    capability_id: str
    profile_id: str
    records: tuple[SuccessorRuntimeEvidenceRecordV1, ...]
    evidence_id: str
    claim_ceiling: str


@dataclass(frozen=True)
class SuccessorRuntimeEvidenceDecisionV1:
    declaration_id: str
    evidence_id: str
    runtime_ready: bool
    satisfied_targets: tuple[str, ...]
    unsatisfied_targets: tuple[str, ...]
    reasons: tuple[str, ...]
    claim_ceiling: str = RUNTIME_EVIDENCE_CLAIM_CEILING


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
            "runtime evidence is not canonicalizable"
        ) from exc


def _strict_mapping(
    value: Any,
    expected_keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the runtime evidence contract"
        )
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ReferenceMatchContractError(
            f"{label} must be a bounded non-empty string"
        )
    return value


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return text


def _commit(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise ReferenceMatchContractError(
            f"{label} must be a full lowercase Git commit"
        )
    return text


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ReferenceMatchContractError(f"{label} must be boolean")
    return value


def runtime_record_id_v1(value: Mapping[str, Any]) -> str:
    if "record_id" in value:
        raise ReferenceMatchContractError(
            "record_id must be omitted while computing identity"
        )
    return hashlib.sha256(
        b"NeuroFilmReferenceMatchSuccessorRuntimeRecordV1\0"
        + _canonical_json(value)
    ).hexdigest()


def runtime_evidence_id_v1(value: Mapping[str, Any]) -> str:
    if "evidence_id" in value:
        raise ReferenceMatchContractError(
            "evidence_id must be omitted while computing identity"
        )
    return hashlib.sha256(
        b"NeuroFilmReferenceMatchSuccessorRuntimeEvidenceV1\0"
        + _canonical_json(value)
    ).hexdigest()


def make_successor_runtime_record_v1(
    *,
    target_runtime: str,
    proof_class: str,
    os_name: str,
    os_version: str,
    architecture: str,
    environment_count: int,
    environment_matrix_sha256: str,
    environment_summary: str,
    backend_id: str,
    backend_version: str,
    runner_sha256: str,
    executable_sha256: str,
    report_sha256: str,
    replay_count: int,
    deterministic_replay_passed: bool,
    conformance_passed: bool,
    failure_injection_passed: bool,
) -> SuccessorRuntimeEvidenceRecordV1:
    value: dict[str, Any] = {
        "target_runtime": target_runtime,
        "proof_class": proof_class,
        "os_name": os_name,
        "os_version": os_version,
        "architecture": architecture,
        "environment_count": environment_count,
        "environment_matrix_sha256": environment_matrix_sha256,
        "environment_summary": environment_summary,
        "backend_id": backend_id,
        "backend_version": backend_version,
        "runner_sha256": runner_sha256,
        "executable_sha256": executable_sha256,
        "report_sha256": report_sha256,
        "replay_count": replay_count,
        "deterministic_replay_passed": deterministic_replay_passed,
        "conformance_passed": conformance_passed,
        "failure_injection_passed": failure_injection_passed,
    }
    value["record_id"] = runtime_record_id_v1(value)
    return _record_from_mapping(value)


def make_successor_runtime_evidence_v1(
    *,
    declaration_id: str,
    producer_commit: str,
    capability_id: str,
    profile_id: str,
    records: Sequence[SuccessorRuntimeEvidenceRecordV1],
) -> SuccessorRuntimeEvidenceV1:
    value: dict[str, Any] = {
        "schema_id": RUNTIME_EVIDENCE_SCHEMA_ID,
        "policy_id": RUNTIME_EVIDENCE_POLICY_ID,
        "declaration_id": declaration_id,
        "producer_commit": producer_commit,
        "capability_id": capability_id,
        "profile_id": profile_id,
        "records": [asdict(record) for record in records],
        "claim_ceiling": RUNTIME_EVIDENCE_CLAIM_CEILING,
    }
    value["evidence_id"] = runtime_evidence_id_v1(value)
    return successor_runtime_evidence_from_dict(value)


def _record_from_mapping(
    value: Mapping[str, Any],
) -> SuccessorRuntimeEvidenceRecordV1:
    record = dict(_strict_mapping(value, _RECORD_KEYS, "runtime record"))
    target = _string(record["target_runtime"], "record.target_runtime")
    if target not in TARGET_RUNTIMES:
        raise ReferenceMatchContractError(
            "record.target_runtime is unsupported"
        )
    proof = _string(record["proof_class"], "record.proof_class")
    if proof not in RUNTIME_PROOF_CLASSES:
        raise ReferenceMatchContractError(
            "record.proof_class is unsupported"
        )
    for key in (
        "os_name",
        "os_version",
        "architecture",
        "environment_summary",
        "backend_id",
        "backend_version",
    ):
        _string(record[key], f"record.{key}")
    environment_count = record["environment_count"]
    if (
        isinstance(environment_count, bool)
        or not isinstance(environment_count, int)
        or not 1 <= environment_count <= 64
    ):
        raise ReferenceMatchContractError(
            "record.environment_count must be an integer in [1,64]"
        )
    for key in (
        "environment_matrix_sha256",
        "runner_sha256",
        "executable_sha256",
        "report_sha256",
    ):
        _sha256(record[key], f"record.{key}")
    replay_count = record["replay_count"]
    if (
        isinstance(replay_count, bool)
        or not isinstance(replay_count, int)
        or not 1 <= replay_count <= 1000
    ):
        raise ReferenceMatchContractError(
            "record.replay_count must be an integer in [1,1000]"
        )
    for key in (
        "deterministic_replay_passed",
        "conformance_passed",
        "failure_injection_passed",
    ):
        _boolean(record[key], f"record.{key}")
    claimed = _sha256(record["record_id"], "record.record_id")
    identity = dict(record)
    identity.pop("record_id")
    if claimed != runtime_record_id_v1(identity):
        raise ReferenceMatchContractError(
            "runtime record identity mismatch"
        )
    return SuccessorRuntimeEvidenceRecordV1(**record)


def successor_runtime_evidence_from_dict(
    value: Mapping[str, Any],
) -> SuccessorRuntimeEvidenceV1:
    bundle = dict(_strict_mapping(value, _BUNDLE_KEYS, "runtime evidence"))
    if (
        bundle["schema_id"] != RUNTIME_EVIDENCE_SCHEMA_ID
        or bundle["policy_id"] != RUNTIME_EVIDENCE_POLICY_ID
    ):
        raise ReferenceMatchContractError(
            "runtime evidence schema/policy is unsupported"
        )
    _sha256(bundle["declaration_id"], "declaration_id")
    _commit(bundle["producer_commit"], "producer_commit")
    _string(bundle["capability_id"], "capability_id")
    _string(bundle["profile_id"], "profile_id")
    raw_records = bundle["records"]
    if not isinstance(raw_records, list) or not 1 <= len(raw_records) <= len(
        TARGET_RUNTIMES
    ):
        raise ReferenceMatchContractError(
            "runtime evidence must contain one to four records"
        )
    records = tuple(_record_from_mapping(record) for record in raw_records)
    targets = tuple(record.target_runtime for record in records)
    if len(set(targets)) != len(targets):
        raise ReferenceMatchContractError(
            "runtime evidence target is duplicated"
        )
    expected_order = tuple(target for target in TARGET_RUNTIMES if target in targets)
    if targets != expected_order:
        raise ReferenceMatchContractError(
            "runtime evidence records are not in canonical target order"
        )
    if bundle["claim_ceiling"] != RUNTIME_EVIDENCE_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "runtime evidence claim ceiling is invalid"
        )
    claimed = _sha256(bundle["evidence_id"], "evidence_id")
    identity = dict(bundle)
    identity.pop("evidence_id")
    if claimed != runtime_evidence_id_v1(identity):
        raise ReferenceMatchContractError(
            "runtime evidence identity mismatch"
        )
    return SuccessorRuntimeEvidenceV1(
        schema_id=bundle["schema_id"],
        policy_id=bundle["policy_id"],
        declaration_id=bundle["declaration_id"],
        producer_commit=bundle["producer_commit"],
        capability_id=bundle["capability_id"],
        profile_id=bundle["profile_id"],
        records=records,
        evidence_id=claimed,
        claim_ceiling=bundle["claim_ceiling"],
    )


def successor_runtime_evidence_to_dict(
    value: SuccessorRuntimeEvidenceV1,
) -> dict[str, Any]:
    return {
        "schema_id": value.schema_id,
        "policy_id": value.policy_id,
        "declaration_id": value.declaration_id,
        "producer_commit": value.producer_commit,
        "capability_id": value.capability_id,
        "profile_id": value.profile_id,
        "records": [asdict(record) for record in value.records],
        "evidence_id": value.evidence_id,
        "claim_ceiling": value.claim_ceiling,
    }


def successor_runtime_evidence_from_json(
    value: str,
) -> SuccessorRuntimeEvidenceV1:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "runtime evidence JSON is invalid"
        ) from exc
    return successor_runtime_evidence_from_dict(parsed)


def successor_runtime_evidence_to_json(
    value: SuccessorRuntimeEvidenceV1,
) -> str:
    return json.dumps(
        successor_runtime_evidence_to_dict(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def validate_successor_runtime_evidence_v1(
    value: SuccessorRuntimeEvidenceV1 | Mapping[str, Any],
) -> SuccessorRuntimeEvidenceV1:
    return (
        successor_runtime_evidence_from_dict(
            successor_runtime_evidence_to_dict(value)
        )
        if isinstance(value, SuccessorRuntimeEvidenceV1)
        else successor_runtime_evidence_from_dict(value)
    )


def bind_successor_runtime_evidence_v1(
    declaration: Mapping[str, Any],
    evidence: SuccessorRuntimeEvidenceV1 | Mapping[str, Any],
) -> SuccessorRuntimeEvidenceDecisionV1:
    admission = evaluate_successor_declaration_v1(declaration)
    bundle = validate_successor_runtime_evidence_v1(evidence)
    producer = declaration["producer"]
    wire = declaration["wire"]
    runtime_claims = declaration["runtime_evidence"]
    if not isinstance(producer, Mapping) or not isinstance(wire, Mapping):
        raise ReferenceMatchContractError(
            "successor declaration binding sections are invalid"
        )
    if not isinstance(runtime_claims, Mapping):
        raise ReferenceMatchContractError(
            "successor runtime claims are invalid"
        )
    expected = {
        "declaration_id": admission.declaration_id,
        "producer_commit": producer["stable_commit"],
        "capability_id": wire["capability_id"],
        "profile_id": wire["profile_id"],
    }
    actual = {
        "declaration_id": bundle.declaration_id,
        "producer_commit": bundle.producer_commit,
        "capability_id": bundle.capability_id,
        "profile_id": bundle.profile_id,
    }
    if actual != expected:
        raise ReferenceMatchContractError(
            "runtime evidence does not bind the successor declaration"
        )

    by_target = {record.target_runtime: record for record in bundle.records}
    satisfied: list[str] = []
    unsatisfied: list[str] = []
    reasons: list[str] = []
    for target in TARGET_RUNTIMES:
        target_reasons: list[str] = []
        if runtime_claims[target] is not True:
            target_reasons.append(f"{target}-declaration-claim-absent")
        record = by_target.get(target)
        if record is None:
            target_reasons.append(f"{target}-evidence-missing")
        else:
            required_class = REQUIRED_RUNTIME_PROOF_CLASS[target]
            if record.proof_class != required_class:
                target_reasons.append(
                    f"{target}-proof-class-{record.proof_class}"
                    f"-not-{required_class}"
                )
            if record.replay_count < 2:
                target_reasons.append(f"{target}-replay-count-insufficient")
            if not record.deterministic_replay_passed:
                target_reasons.append(
                    f"{target}-deterministic-replay-failed"
                )
            if not record.conformance_passed:
                target_reasons.append(f"{target}-conformance-failed")
            if not record.failure_injection_passed:
                target_reasons.append(
                    f"{target}-failure-injection-failed"
                )
        if target_reasons:
            unsatisfied.append(target)
            reasons.extend(target_reasons)
        else:
            satisfied.append(target)
    return SuccessorRuntimeEvidenceDecisionV1(
        declaration_id=admission.declaration_id,
        evidence_id=bundle.evidence_id,
        runtime_ready=not reasons,
        satisfied_targets=tuple(satisfied),
        unsatisfied_targets=tuple(unsatisfied),
        reasons=tuple(reasons),
    )


__all__ = [
    "REQUIRED_RUNTIME_PROOF_CLASS",
    "RUNTIME_EVIDENCE_CLAIM_CEILING",
    "RUNTIME_EVIDENCE_POLICY_ID",
    "RUNTIME_EVIDENCE_SCHEMA_ID",
    "RUNTIME_PROOF_CLASSES",
    "SuccessorRuntimeEvidenceDecisionV1",
    "SuccessorRuntimeEvidenceRecordV1",
    "SuccessorRuntimeEvidenceV1",
    "bind_successor_runtime_evidence_v1",
    "make_successor_runtime_evidence_v1",
    "make_successor_runtime_record_v1",
    "runtime_evidence_id_v1",
    "runtime_record_id_v1",
    "successor_runtime_evidence_from_dict",
    "successor_runtime_evidence_from_json",
    "successor_runtime_evidence_to_dict",
    "successor_runtime_evidence_to_json",
    "validate_successor_runtime_evidence_v1",
]
