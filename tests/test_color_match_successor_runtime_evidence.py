from __future__ import annotations

from dataclasses import asdict, replace
import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.successor_admission import (
    FROZEN_GATE_POLICY_ID,
    SUCCESSOR_DECLARATION_SCHEMA,
    SUCCESSOR_POLICY_ID,
    successor_declaration_id_v1,
)
from src.color_match.successor_runtime_evidence import (
    REQUIRED_RUNTIME_PROOF_CLASS,
    RUNTIME_EVIDENCE_CLAIM_CEILING,
    RUNTIME_EVIDENCE_POLICY_ID,
    RUNTIME_EVIDENCE_SCHEMA_ID,
    SuccessorRuntimeEvidenceRecordV1,
    bind_successor_runtime_evidence_v1,
    make_successor_runtime_evidence_v1,
    make_successor_runtime_record_v1,
    runtime_evidence_id_v1,
    runtime_record_id_v1,
    successor_runtime_evidence_from_dict,
    successor_runtime_evidence_from_json,
    successor_runtime_evidence_to_dict,
    successor_runtime_evidence_to_json,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_match_successor_runtime_evidence_v1.schema.json"
)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
PRODUCER_COMMIT = "1" * 40
CAPABILITY_ID = "zhuise.future-reference-shared.cpu.v2"
PROFILE_ID = "zhuise.display-linear-srgb-d65-relative-f32.v1"


def _declaration(*, runtime_claims: bool = True) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": SUCCESSOR_DECLARATION_SCHEMA,
        "policy_id": SUCCESSOR_POLICY_ID,
        "candidate_id": "zhuise.future-reference-shared.v2",
        "producer": {
            "stable_commit": PRODUCER_COMMIT,
            "package_source_commit": "2" * 40,
            "fixture_commit": "3" * 40,
            "package_lock_sha256": SHA_A,
            "conformance_fixture_sha256": SHA_B,
        },
        "package": {
            "distribution": "zhuise-research",
            "version": "0.4.0",
            "entrypoint": "zhuise-producer-invoke",
            "wheel_filename": "zhuise_research-0.4.0-py3-none-any.whl",
            "wheel_size_bytes": 123456,
            "wheel_sha256": SHA_C,
        },
        "wire": {
            "capability_id": CAPABILITY_ID,
            "profile_id": PROFILE_ID,
            "lower_compatibility_profile_id": "neuro-film.dpct-consumer.v2",
            "request_schema": "zhuise.invocation-request.v2",
            "request_schema_sha256": SHA_A,
            "response_schema": "zhuise.invocation-response.v2",
            "response_schema_sha256": SHA_B,
        },
        "semantics": {
            "fit_semantics": "reference-only-shared",
            "batch_transform_policy": "shared-bundle",
            "deterministic": True,
            "hidden_state": "none",
        },
        "rights": {
            "evaluation_allowed": True,
            "commercial_use_allowed": True,
            "redistribution_allowed": True,
            "evidence_id": "rights-review-v2",
        },
        "runtime_evidence": {
            "windows_x64": runtime_claims,
            "macos_arm64": runtime_claims,
            "ios_arm64": runtime_claims,
            "android_arm64": runtime_claims,
        },
        "product_evidence": {
            "gate_policy_id": FROZEN_GATE_POLICY_ID,
            "stable_evidence_id": SHA_D,
            "a1_passed": True,
            "a4_passed": True,
            "a5_passed": True,
            "blind_aesthetic_passed": True,
        },
    }
    value["declaration_id"] = successor_declaration_id_v1(value)
    return value


def _record(
    target: str,
    *,
    proof_class: str | None = None,
    replay_count: int = 2,
    deterministic: bool = True,
    conformance: bool = True,
    failure_injection: bool = True,
) -> SuccessorRuntimeEvidenceRecordV1:
    architecture = {
        "windows_x64": "x86_64",
        "macos_arm64": "arm64",
        "ios_arm64": "arm64",
        "android_arm64": "arm64-v8a",
    }[target]
    return make_successor_runtime_record_v1(
        target_runtime=target,
        proof_class=proof_class or REQUIRED_RUNTIME_PROOF_CLASS[target],
        os_name={
            "windows_x64": "Windows",
            "macos_arm64": "macOS",
            "ios_arm64": "iOS",
            "android_arm64": "Android",
        }[target],
        os_version="test-os-1",
        architecture=architecture,
        environment_count=2 if target == "windows_x64" else 1,
        environment_matrix_sha256=SHA_D,
        environment_summary=(
            "test-nvidia-and-amd-matrix"
            if target == "windows_x64"
            else f"test-{target}-device"
        ),
        backend_id="test-backend",
        backend_version="test-backend-1",
        runner_sha256=SHA_A,
        executable_sha256=SHA_B,
        report_sha256=SHA_C,
        replay_count=replay_count,
        deterministic_replay_passed=deterministic,
        conformance_passed=conformance,
        failure_injection_passed=failure_injection,
    )


def _all_records() -> tuple[SuccessorRuntimeEvidenceRecordV1, ...]:
    return (
        _record("windows_x64"),
        _record("macos_arm64"),
        _record("ios_arm64"),
        _record("android_arm64"),
    )


def _bundle(
    declaration: dict[str, object],
    records: tuple[SuccessorRuntimeEvidenceRecordV1, ...],
):
    return make_successor_runtime_evidence_v1(
        declaration_id=str(declaration["declaration_id"]),
        producer_commit=PRODUCER_COMMIT,
        capability_id=CAPABILITY_ID,
        profile_id=PROFILE_ID,
        records=records,
    )


def test_complete_runtime_bundle_is_strict_roundtrippable_and_ready() -> None:
    declaration = _declaration()
    bundle = _bundle(declaration, _all_records())
    encoded = successor_runtime_evidence_to_json(bundle)
    assert successor_runtime_evidence_from_json(encoded) == bundle
    payload = successor_runtime_evidence_to_dict(bundle)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    decision = bind_successor_runtime_evidence_v1(declaration, bundle)
    assert decision.runtime_ready
    assert decision.satisfied_targets == (
        "windows_x64",
        "macos_arm64",
        "ios_arm64",
        "android_arm64",
    )
    assert decision.unsatisfied_targets == ()
    assert decision.reasons == ()
    assert decision.claim_ceiling == RUNTIME_EVIDENCE_CLAIM_CEILING


def test_windows_only_runtime_evidence_does_not_close_other_targets() -> None:
    declaration = _declaration()
    decision = bind_successor_runtime_evidence_v1(
        declaration,
        _bundle(declaration, (_record("windows_x64"),)),
    )
    assert not decision.runtime_ready
    assert decision.satisfied_targets == ("windows_x64",)
    assert decision.unsatisfied_targets == (
        "macos_arm64",
        "ios_arm64",
        "android_arm64",
    )
    assert decision.reasons == (
        "macos_arm64-evidence-missing",
        "ios_arm64-evidence-missing",
        "android_arm64-evidence-missing",
    )


@pytest.mark.parametrize(
    ("target", "proof", "required"),
    [
        ("windows_x64", "cross-compiled", "host-runtime"),
        ("macos_arm64", "object-only", "host-runtime"),
        ("ios_arm64", "link-only", "device-runtime"),
        ("android_arm64", "cross-compiled", "device-runtime"),
    ],
)
def test_weak_proof_class_cannot_close_runtime(
    target: str,
    proof: str,
    required: str,
) -> None:
    declaration = _declaration()
    decision = bind_successor_runtime_evidence_v1(
        declaration,
        _bundle(declaration, (_record(target, proof_class=proof),)),
    )
    assert not decision.runtime_ready
    assert (
        f"{target}-proof-class-{proof}-not-{required}"
        in decision.reasons
    )


def test_replay_and_failure_facts_are_independent_fail_closed_gates() -> None:
    declaration = _declaration()
    weak = _record(
        "windows_x64",
        replay_count=1,
        deterministic=False,
        conformance=False,
        failure_injection=False,
    )
    decision = bind_successor_runtime_evidence_v1(
        declaration,
        _bundle(declaration, (weak,)),
    )
    assert set(decision.reasons[:4]) == {
        "windows_x64-replay-count-insufficient",
        "windows_x64-deterministic-replay-failed",
        "windows_x64-conformance-failed",
        "windows_x64-failure-injection-failed",
    }


def test_runtime_evidence_does_not_override_false_declaration_claim() -> None:
    declaration = _declaration(runtime_claims=False)
    decision = bind_successor_runtime_evidence_v1(
        declaration,
        _bundle(declaration, _all_records()),
    )
    assert not decision.runtime_ready
    assert decision.satisfied_targets == ()
    assert all(
        f"{target}-declaration-claim-absent" in decision.reasons
        for target in REQUIRED_RUNTIME_PROOF_CLASS
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("declaration_id", "f" * 64),
        ("producer_commit", "f" * 40),
        ("capability_id", "foreign-capability"),
        ("profile_id", "foreign-profile"),
    ],
)
def test_foreign_declaration_or_wire_binding_rejects(
    field: str,
    value: str,
) -> None:
    declaration = _declaration()
    bundle = _bundle(declaration, (_record("windows_x64"),))
    foreign = replace(bundle, **{field: value})
    raw = successor_runtime_evidence_to_dict(foreign)
    raw.pop("evidence_id")
    raw["evidence_id"] = runtime_evidence_id_v1(raw)
    with pytest.raises(
        ReferenceMatchContractError,
        match="does not bind",
    ):
        bind_successor_runtime_evidence_v1(declaration, raw)


def test_duplicate_and_noncanonical_target_order_reject() -> None:
    declaration = _declaration()
    with pytest.raises(
        ReferenceMatchContractError,
        match="duplicated",
    ):
        _bundle(
            declaration,
            (_record("windows_x64"), _record("windows_x64")),
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="canonical target order",
    ):
        _bundle(
            declaration,
            (_record("android_arm64"), _record("windows_x64")),
        )


@pytest.mark.parametrize(
    ("target", "field", "foreign_value"),
    [
        ("windows_x64", "os_name", "iOS"),
        ("macos_arm64", "architecture", "x86_64"),
        ("ios_arm64", "os_name", "Windows"),
        ("android_arm64", "architecture", "arm64"),
    ],
)
def test_target_os_architecture_contradiction_rejects_even_when_rehashed(
    target: str,
    field: str,
    foreign_value: str,
) -> None:
    declaration = _declaration()
    payload = successor_runtime_evidence_to_dict(
        _bundle(declaration, (_record(target),))
    )
    record = payload["records"][0]
    record[field] = foreign_value
    record.pop("record_id")
    record["record_id"] = runtime_record_id_v1(record)
    payload.pop("evidence_id")
    payload["evidence_id"] = runtime_evidence_id_v1(payload)
    with pytest.raises(
        ReferenceMatchContractError,
        match="target/os/architecture mapping",
    ):
        successor_runtime_evidence_from_dict(payload)


def test_record_and_bundle_identity_tamper_reject() -> None:
    declaration = _declaration()
    original = _bundle(declaration, (_record("windows_x64"),))
    payload = successor_runtime_evidence_to_dict(original)
    record_tamper = copy.deepcopy(payload)
    record_tamper["records"][0]["environment_summary"] = "tampered"
    with pytest.raises(
        ReferenceMatchContractError,
        match="record identity mismatch",
    ):
        successor_runtime_evidence_from_dict(record_tamper)

    bundle_tamper = copy.deepcopy(payload)
    bundle_tamper["claim_ceiling"] = "product-ready"
    with pytest.raises(
        ReferenceMatchContractError,
        match="claim ceiling",
    ):
        successor_runtime_evidence_from_dict(bundle_tamper)

    forged_record = replace(original.records[0], replay_count=1)
    forged_dataclass = replace(original, records=(forged_record,))
    with pytest.raises(
        ReferenceMatchContractError,
        match="record identity mismatch",
    ):
        bind_successor_runtime_evidence_v1(
            declaration,
            forged_dataclass,
        )


def test_identity_helpers_require_omitted_identity_fields() -> None:
    record = asdict(_record("windows_x64"))
    with pytest.raises(
        ReferenceMatchContractError,
        match="record_id must be omitted",
    ):
        runtime_record_id_v1(record)
    declaration = _declaration()
    bundle = successor_runtime_evidence_to_dict(
        _bundle(declaration, (_record("windows_x64"),))
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="evidence_id must be omitted",
    ):
        runtime_evidence_id_v1(bundle)


def test_unknown_fields_nonfinite_json_and_invalid_hash_reject() -> None:
    declaration = _declaration()
    payload = successor_runtime_evidence_to_dict(
        _bundle(declaration, (_record("windows_x64"),))
    )
    payload["unknown"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        successor_runtime_evidence_from_dict(payload)

    with pytest.raises(ReferenceMatchContractError, match="JSON is invalid"):
        successor_runtime_evidence_from_json("{")

    bad = asdict(_record("windows_x64"))
    bad["runner_sha256"] = "not-a-hash"
    bad.pop("record_id")
    bad["record_id"] = runtime_record_id_v1(bad)
    with pytest.raises(ReferenceMatchContractError, match="lowercase SHA-256"):
        successor_runtime_evidence_from_dict(
            {
                **successor_runtime_evidence_to_dict(
                    _bundle(declaration, (_record("windows_x64"),))
                ),
                "records": [bad],
            }
        )


def test_schema_is_strict_and_ids_match_implementation() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_id"]["const"] == (
        RUNTIME_EVIDENCE_SCHEMA_ID
    )
    assert schema["properties"]["policy_id"]["const"] == (
        RUNTIME_EVIDENCE_POLICY_ID
    )
    assert schema["properties"]["claim_ceiling"]["const"] == (
        RUNTIME_EVIDENCE_CLAIM_CEILING
    )
    assert schema["$defs"]["runtime_record"]["additionalProperties"] is False
