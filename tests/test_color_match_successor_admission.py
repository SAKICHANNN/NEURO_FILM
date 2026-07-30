from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.successor_admission import (
    FROZEN_GATE_POLICY_ID,
    REJECTED_CAPABILITY_ID,
    REJECTED_PRODUCER_STABLE_COMMIT,
    REJECTED_STABLE_EVIDENCE_ID,
    REJECTED_WHEEL_SHA256,
    SUCCESSOR_DECLARATION_SCHEMA,
    SUCCESSOR_POLICY_ID,
    evaluate_successor_declaration_v1,
    successor_declaration_id_v1,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "reference_match_successor_admission_v1.json"
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_match_successor_declaration_v1.schema.json"
)
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _declaration(*, product: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": SUCCESSOR_DECLARATION_SCHEMA,
        "policy_id": SUCCESSOR_POLICY_ID,
        "candidate_id": "zhuise.future-candidate.v2",
        "producer": {
            "stable_commit": "1" * 40,
            "package_source_commit": "2" * 40,
            "fixture_commit": "3" * 40,
            "package_lock_sha256": HEX_A,
            "conformance_fixture_sha256": HEX_B,
        },
        "package": {
            "distribution": "zhuise-research",
            "version": "0.3.0",
            "entrypoint": "zhuise-producer-invoke",
            "wheel_filename": "zhuise_research-0.3.0-py3-none-any.whl",
            "wheel_size_bytes": 100000,
            "wheel_sha256": HEX_C,
        },
        "wire": {
            "capability_id": "zhuise.future-candidate.cpu-reference.v2",
            "profile_id": (
                "zhuise.display-linear-srgb-d65-relative-f32.v1"
            ),
            "lower_compatibility_profile_id": (
                "neuro-film.dpct-consumer.v2"
            ),
            "request_schema": "zhuise.invocation-request.v2",
            "request_schema_sha256": HEX_A,
            "response_schema": "zhuise.invocation-response.v2",
            "response_schema_sha256": HEX_B,
        },
        "semantics": {
            "fit_semantics": "source-reference-per-source",
            "batch_transform_policy": "per-source-bundle",
            "deterministic": True,
            "hidden_state": "none",
        },
        "rights": {
            "evaluation_allowed": True,
            "commercial_use_allowed": product,
            "redistribution_allowed": product,
            "evidence_id": "rights-review-v1",
        },
        "runtime_evidence": {
            "windows_x64": product,
            "macos_arm64": product,
            "ios_arm64": product,
            "android_arm64": product,
        },
        "product_evidence": (
            {
                "gate_policy_id": FROZEN_GATE_POLICY_ID,
                "stable_evidence_id": HEX_C,
                "a1_passed": True,
                "a4_passed": True,
                "a5_passed": True,
                "blind_aesthetic_passed": True,
            }
            if product
            else None
        ),
    }
    value["declaration_id"] = successor_declaration_id_v1(value)
    return value


def _resign(value: dict[str, object]) -> dict[str, object]:
    value.pop("declaration_id", None)
    value["declaration_id"] = successor_declaration_id_v1(value)
    return value


def test_policy_freezes_p44_thresholds_and_rejected_identity() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    assert policy["rejected_candidate"] == {
        "capability_id": REJECTED_CAPABILITY_ID,
        "wheel_sha256": REJECTED_WHEEL_SHA256,
        "producer_stable_commit": REJECTED_PRODUCER_STABLE_COMMIT,
        "stable_evidence_id": REJECTED_STABLE_EVIDENCE_ID,
    }
    assert policy["product_readiness"] == {
        "gate_policy_id": FROZEN_GATE_POLICY_ID,
        "minimum_known_operator_samples": 12,
        "minimum_cross_content_improvement_rate": 0.75,
        "minimum_median_improvement_fraction": 0.1,
        "minimum_worst_improvement_fraction": -0.1,
        "maximum_new_boundary_fraction": 0.05,
        "maximum_neutral_chroma_p95": 18.0,
        "maximum_tone_reversal_fraction": 0.0,
        "maximum_tone_plateau_fraction": 0.2,
        "maximum_semantic_hue_rotation_p95_degrees": 75.0,
        "maximum_context_median_delta_e76": 0.5,
        "maximum_context_p95_delta_e76": 1.0,
        "maximum_context_delta_e76": 3.0,
        "require_a1": True,
        "require_a4": True,
        "require_a5": True,
        "minimum_blind_review_samples": 12,
        "minimum_blind_preference_rate": 0.5,
        "maximum_blind_severe_artifact_count": 0,
        "require_commercial_rights": True,
        "require_redistribution_rights": True,
        "required_target_runtimes": [
            "windows_x64",
            "macos_arm64",
            "ios_arm64",
            "android_arm64",
        ],
    }


def test_declaration_schema_and_canonical_identity() -> None:
    value = _declaration()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(value)
    decision = evaluate_successor_declaration_v1(value)
    assert decision.evaluation_ready
    assert not decision.product_ready
    assert decision.evaluation_reasons == ()
    assert set(decision.product_reasons) == {
        "product-evidence-absent",
        "commercial-rights-absent",
        "redistribution-rights-absent",
        "windows_x64-runtime-absent",
        "macos_arm64-runtime-absent",
        "ios_arm64-runtime-absent",
        "android_arm64-runtime-absent",
    }

    tampered = copy.deepcopy(value)
    tampered["candidate_id"] = "different"
    with pytest.raises(
        ReferenceMatchContractError,
        match="identity mismatch",
    ):
        evaluate_successor_declaration_v1(tampered)


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (
            ("wire", "capability_id"),
            REJECTED_CAPABILITY_ID,
            "reuses-rejected-capability",
        ),
        (
            ("package", "wheel_sha256"),
            REJECTED_WHEEL_SHA256,
            "reuses-rejected-wheel",
        ),
        (
            ("producer", "stable_commit"),
            REJECTED_PRODUCER_STABLE_COMMIT,
            "reuses-rejected-producer-snapshot",
        ),
        (
            ("wire", "profile_id"),
            "zhuise.display-linear-bt2020-d65-absolute-nits-f32.v2",
            "unsupported-colour-profile",
        ),
        (
            ("wire", "lower_compatibility_profile_id"),
            "neuro-film.dpct-consumer.v3",
            "unsupported-lower-compatibility-profile",
        ),
        (
            ("rights", "evaluation_allowed"),
            False,
            "evaluation-rights-absent",
        ),
    ],
)
def test_evaluation_preflight_rejects_substitution(
    path: tuple[str, str],
    value: object,
    reason: str,
) -> None:
    declaration = _declaration()
    section = declaration[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value
    decision = evaluate_successor_declaration_v1(_resign(declaration))
    assert not decision.evaluation_ready
    assert reason in decision.evaluation_reasons
    assert reason in decision.product_reasons


def test_fit_and_batch_semantics_must_agree() -> None:
    declaration = _declaration()
    semantics = declaration["semantics"]
    assert isinstance(semantics, dict)
    semantics["batch_transform_policy"] = "shared-bundle"
    with pytest.raises(
        ReferenceMatchContractError,
        match="semantics are inconsistent",
    ):
        evaluate_successor_declaration_v1(_resign(declaration))


def test_product_ready_requires_new_p44_evidence_rights_and_runtime() -> None:
    decision = evaluate_successor_declaration_v1(_declaration(product=True))
    assert decision.evaluation_ready
    assert decision.product_ready
    assert decision.product_reasons == ()

    declaration = _declaration(product=True)
    evidence = declaration["product_evidence"]
    assert isinstance(evidence, dict)
    evidence["stable_evidence_id"] = REJECTED_STABLE_EVIDENCE_ID
    evidence["a5_passed"] = False
    rights = declaration["rights"]
    assert isinstance(rights, dict)
    rights["commercial_use_allowed"] = False
    runtimes = declaration["runtime_evidence"]
    assert isinstance(runtimes, dict)
    runtimes["ios_arm64"] = False
    decision = evaluate_successor_declaration_v1(_resign(declaration))
    assert decision.evaluation_ready
    assert not decision.product_ready
    assert set(decision.product_reasons) == {
        "reuses-rejected-product-evidence",
        "a5-not-passed",
        "commercial-rights-absent",
        "ios_arm64-runtime-absent",
    }


def test_strict_fields_and_finite_json_fail_closed() -> None:
    declaration = _declaration()
    declaration["unknown"] = True
    with pytest.raises(
        ReferenceMatchContractError,
        match="fields differ",
    ):
        evaluate_successor_declaration_v1(declaration)

    declaration = _declaration()
    package = declaration["package"]
    assert isinstance(package, dict)
    package["version"] = math.nan
    with pytest.raises(ReferenceMatchContractError):
        evaluate_successor_declaration_v1(_resign(declaration))
