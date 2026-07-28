from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from src.color_match import (
    FROZEN_GATE_POLICY_ID,
    MATCH_PROFILE_DISPLAY_SRGB,
    SHARED_RUNTIME_QUALIFICATION_CLAIM_CEILING,
    SHARED_RUNTIME_QUALIFICATION_SCHEMA_ID,
    SUCCESSOR_DECLARATION_SCHEMA,
    SUCCESSOR_POLICY_ID,
    SUPPORTED_PROFILE_ID,
    PreparedMatchViewV1,
    PromotionDecision,
    ReferenceMatchContractError,
    authorize_shared_product_staging_v1,
    bind_shared_promotion_v1,
    guard_shared_numeric_batch_v1,
    make_match_view,
    make_shared_apply_numeric_facts_v1,
    make_shared_reference_operator_v1,
    make_successor_runtime_evidence_v1,
    make_successor_runtime_record_v1,
    prepare_shared_operator_apply_v1,
    qualify_shared_product_authorization_runtime_v1,
    resolve_shared_operator_batch_v1,
    runtime_evidence_id_v1,
    runtime_qualified_shared_authorization_from_dict,
    runtime_qualified_shared_authorization_from_json,
    runtime_qualified_shared_authorization_to_dict,
    runtime_qualified_shared_authorization_to_json,
    successor_declaration_id_v1,
    successor_runtime_evidence_to_dict,
    validate_runtime_qualified_shared_authorization_binding_v1,
    validate_runtime_qualified_shared_authorization_v1,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_shared_runtime_qualification_v1.schema.json"
)
CAPABILITY = "zhuise.runtime-qualified.shared.v1"
PRODUCER_COMMIT = "1" * 40
MODEL = "d" * 64
OPTIONS = "e" * 64
EVIDENCE = "f" * 64
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _prepared(seed: float) -> PreparedMatchViewV1:
    pixels = np.full((2, 3, 3), seed, dtype=np.float32)
    wire = pixels.astype(">f4", copy=False).tobytes()
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=pixels.shape,
        render_bridge_id="neuro-film.runtime-qualified-test.v1",
        provenance_fingerprint=hashlib.sha256(
            str(seed).encode("ascii")
        ).hexdigest(),
    )
    pixels.flags.writeable = False
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _declaration() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": SUCCESSOR_DECLARATION_SCHEMA,
        "policy_id": SUCCESSOR_POLICY_ID,
        "candidate_id": "zhuise.runtime-qualified.shared.v1",
        "producer": {
            "stable_commit": PRODUCER_COMMIT,
            "package_source_commit": "2" * 40,
            "fixture_commit": "3" * 40,
            "package_lock_sha256": SHA_A,
            "conformance_fixture_sha256": SHA_B,
        },
        "package": {
            "distribution": "zhuise-research",
            "version": "0.5.0",
            "entrypoint": "zhuise-shared-invoke",
            "wheel_filename": "zhuise_research-0.5.0-py3-none-any.whl",
            "wheel_size_bytes": 120000,
            "wheel_sha256": SHA_C,
        },
        "wire": {
            "capability_id": CAPABILITY,
            "profile_id": SUPPORTED_PROFILE_ID,
            "lower_compatibility_profile_id": (
                "neuro-film.dpct-consumer.v2"
            ),
            "request_schema": "zhuise.shared-request.v2",
            "request_schema_sha256": SHA_A,
            "response_schema": "zhuise.shared-response.v2",
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
            "windows_x64": True,
            "macos_arm64": True,
            "ios_arm64": True,
            "android_arm64": True,
        },
        "product_evidence": {
            "gate_policy_id": FROZEN_GATE_POLICY_ID,
            "stable_evidence_id": EVIDENCE,
            "a1_passed": True,
            "a4_passed": True,
            "a5_passed": True,
            "blind_aesthetic_passed": True,
        },
    }
    value["declaration_id"] = successor_declaration_id_v1(value)
    return value


def _authorization(
    *,
    promoted: bool = True,
    model: str = MODEL,
):
    reference = _prepared(0.2)
    source = _prepared(0.3)
    declaration = _declaration()
    operator = make_shared_reference_operator_v1(
        reference=reference,
        compatibility_profile_id="neuro-film.dpct-consumer.v2",
        capability_id=CAPABILITY,
        producer_commit=PRODUCER_COMMIT,
        producer_bundle_id="sha256:" + "9" * 64,
        producer_reference_view_id="sha256:" + "8" * 64,
        model_fingerprint=model,
        options_sha256=OPTIONS,
    )
    apply = prepare_shared_operator_apply_v1(
        operator=operator,
        source_index=0,
        source=source,
        producer_source_view_id="sha256:" + "1" * 64,
        producer_apply_result_id="sha256:" + "3" * 64,
        diagnostics_id="sha256:" + "5" * 64,
        output_pixels=np.full(
            source.descriptor.shape,
            0.31,
            dtype=np.float32,
        ),
    )
    batch = resolve_shared_operator_batch_v1(
        operator=operator,
        reference=reference,
        sources=(source,),
        applies=(apply,),
    )
    fact = make_shared_apply_numeric_facts_v1(
        prepared=apply,
        producer_diagnostics_id=apply.receipt.diagnostics_id,
        all_finite=True,
        output_minimum=float(np.min(apply.pixels)),
        output_maximum=float(np.max(apply.pixels)),
        out_of_gamut_fraction=0.01,
        clipping_fraction=0.01,
        projected_fraction=0.0,
    )
    numeric = guard_shared_numeric_batch_v1(
        batch=batch,
        sources=(source,),
        applies=(apply,),
        facts=(fact,),
    )
    promotion = (
        PromotionDecision("promoted", ())
        if promoted
        else PromotionDecision(
            "eligible-for-visual-review",
            ("visual-review-required",),
        )
    )
    binding = bind_shared_promotion_v1(
        operator=operator,
        declaration=declaration,
        stable_evidence_id=EVIDENCE,
        promotion=promotion,
    )
    authorization = authorize_shared_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        declaration=declaration,
        promotion_binding=binding,
    )
    return declaration, authorization


def _runtime_record(
    target: str,
    *,
    proof_class: str | None = None,
):
    required = {
        "windows_x64": "host-runtime",
        "macos_arm64": "host-runtime",
        "ios_arm64": "device-runtime",
        "android_arm64": "device-runtime",
    }[target]
    return make_successor_runtime_record_v1(
        target_runtime=target,
        proof_class=proof_class or required,
        os_name=target.split("_", 1)[0],
        os_version="test-os-1",
        architecture=target.rsplit("_", 1)[1],
        environment_count=1,
        environment_matrix_sha256="4" * 64,
        environment_summary=f"one-{target}-environment",
        backend_id="test-backend",
        backend_version="test-backend-1",
        runner_sha256=SHA_A,
        executable_sha256=SHA_B,
        report_sha256=SHA_C,
        replay_count=2,
        deterministic_replay_passed=True,
        conformance_passed=True,
        failure_injection_passed=True,
    )


def _runtime_bundle(
    declaration: dict[str, object],
    targets: tuple[str, ...] = (
        "windows_x64",
        "macos_arm64",
        "ios_arm64",
        "android_arm64",
    ),
):
    return make_successor_runtime_evidence_v1(
        declaration_id=str(declaration["declaration_id"]),
        producer_commit=PRODUCER_COMMIT,
        capability_id=CAPABILITY,
        profile_id=SUPPORTED_PROFILE_ID,
        records=tuple(_runtime_record(target) for target in targets),
    )


def test_exact_p49_and_four_runtime_targets_qualify_staging() -> None:
    declaration, authorization = _authorization()
    result = qualify_shared_product_authorization_runtime_v1(
        authorization=authorization,
        runtime_evidence=_runtime_bundle(declaration),
    )
    assert authorization.state == "authorized-for-staging"
    assert result.runtime_ready
    assert result.state == "runtime-qualified-for-staging"
    assert result.reasons == ()
    assert result.runtime_unsatisfied_targets == ()
    assert result.claim_ceiling == (
        SHARED_RUNTIME_QUALIFICATION_CLAIM_CEILING
    )
    validate_runtime_qualified_shared_authorization_binding_v1(
        result,
        authorization=authorization,
    )


def test_qualification_is_strict_schema_valid_and_roundtrippable() -> None:
    declaration, authorization = _authorization()
    result = qualify_shared_product_authorization_runtime_v1(
        authorization=authorization,
        runtime_evidence=_runtime_bundle(declaration),
    )
    encoded = runtime_qualified_shared_authorization_to_json(result)
    assert runtime_qualified_shared_authorization_from_json(encoded) == result
    payload = runtime_qualified_shared_authorization_to_dict(result)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def test_p49_runtime_booleans_without_four_records_fall_back() -> None:
    declaration, authorization = _authorization()
    result = qualify_shared_product_authorization_runtime_v1(
        authorization=authorization,
        runtime_evidence=_runtime_bundle(
            declaration,
            ("windows_x64",),
        ),
    )
    assert authorization.admission_product_ready
    assert authorization.state == "authorized-for-staging"
    assert not result.runtime_ready
    assert result.state == "identity-fallback"
    assert result.runtime_satisfied_targets == ("windows_x64",)
    assert result.runtime_unsatisfied_targets == (
        "macos_arm64",
        "ios_arm64",
        "android_arm64",
    )
    assert "runtime:ios_arm64-evidence-missing" in result.reasons


def test_runtime_ready_cannot_override_upstream_p49_fallback() -> None:
    declaration, authorization = _authorization(promoted=False)
    result = qualify_shared_product_authorization_runtime_v1(
        authorization=authorization,
        runtime_evidence=_runtime_bundle(declaration),
    )
    assert result.runtime_ready
    assert result.state == "identity-fallback"
    assert result.reasons == ("upstream-authorization-not-ready",)


def test_cross_compile_is_retained_but_not_qualified() -> None:
    declaration, authorization = _authorization()
    records = (
        _runtime_record("windows_x64"),
        _runtime_record("macos_arm64"),
        _runtime_record("ios_arm64"),
        _runtime_record("android_arm64", proof_class="cross-compiled"),
    )
    evidence = make_successor_runtime_evidence_v1(
        declaration_id=str(declaration["declaration_id"]),
        producer_commit=PRODUCER_COMMIT,
        capability_id=CAPABILITY,
        profile_id=SUPPORTED_PROFILE_ID,
        records=records,
    )
    result = qualify_shared_product_authorization_runtime_v1(
        authorization=authorization,
        runtime_evidence=evidence,
    )
    assert not result.runtime_ready
    assert (
        "runtime:android_arm64-proof-class-cross-compiled-not-device-runtime"
        in result.reasons
    )


def test_foreign_p49_authorization_binding_rejects() -> None:
    declaration, authorization = _authorization()
    result = qualify_shared_product_authorization_runtime_v1(
        authorization=authorization,
        runtime_evidence=_runtime_bundle(declaration),
    )
    _other_declaration, foreign = _authorization(model="7" * 64)
    with pytest.raises(
        ReferenceMatchContractError,
        match="does not bind the upstream",
    ):
        validate_runtime_qualified_shared_authorization_binding_v1(
            result,
            authorization=foreign,
        )


def test_foreign_runtime_evidence_declaration_rejects() -> None:
    declaration, authorization = _authorization()
    evidence = successor_runtime_evidence_to_dict(
        _runtime_bundle(declaration)
    )
    evidence["declaration_id"] = "7" * 64
    evidence.pop("evidence_id")
    evidence["evidence_id"] = runtime_evidence_id_v1(evidence)
    with pytest.raises(
        ReferenceMatchContractError,
        match="does not bind",
    ):
        qualify_shared_product_authorization_runtime_v1(
            authorization=authorization,
            runtime_evidence=evidence,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_ready", False, "decision facts"),
        ("upstream_state", "identity-fallback", "state/reasons"),
        ("state", "identity-fallback", "state/reasons"),
        ("qualification_id", "0" * 64, "identity mismatch"),
        ("claim_ceiling", "product-applied", "claim ceiling"),
    ],
)
def test_qualification_fact_and_identity_tamper_reject(
    field: str,
    value: object,
    message: str,
) -> None:
    declaration, authorization = _authorization()
    result = qualify_shared_product_authorization_runtime_v1(
        authorization=authorization,
        runtime_evidence=_runtime_bundle(declaration),
    )
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_runtime_qualified_shared_authorization_v1(
            replace(result, **{field: value})
        )


def test_embedded_runtime_json_tamper_rejects() -> None:
    declaration, authorization = _authorization()
    result = qualify_shared_product_authorization_runtime_v1(
        authorization=authorization,
        runtime_evidence=_runtime_bundle(declaration),
    )
    embedded = json.loads(result.runtime_evidence_json)
    embedded["records"][0]["environment_summary"] = "tampered"
    forged = replace(
        result,
        runtime_evidence_json=json.dumps(
            embedded,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="record identity mismatch",
    ):
        validate_runtime_qualified_shared_authorization_v1(forged)


def test_unknown_fields_and_invalid_json_fail_closed() -> None:
    declaration, authorization = _authorization()
    result = qualify_shared_product_authorization_runtime_v1(
        authorization=authorization,
        runtime_evidence=_runtime_bundle(declaration),
    )
    payload = runtime_qualified_shared_authorization_to_dict(result)
    payload["unknown"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        runtime_qualified_shared_authorization_from_dict(payload)
    payload.pop("unknown")
    payload["reasons"] = False
    with pytest.raises(ReferenceMatchContractError, match="must be an array"):
        runtime_qualified_shared_authorization_from_dict(payload)
    with pytest.raises(ReferenceMatchContractError, match="JSON is invalid"):
        runtime_qualified_shared_authorization_from_json("{")


def test_schema_ids_and_claim_ceiling_are_exact() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_id"]["const"] == (
        SHARED_RUNTIME_QUALIFICATION_SCHEMA_ID
    )
    assert schema["properties"]["claim_ceiling"]["const"] == (
        SHARED_RUNTIME_QUALIFICATION_CLAIM_CEILING
    )
