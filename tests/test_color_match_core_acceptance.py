from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    CORE_ACCEPTANCE_SCHEMA_ID,
    DIAGNOSTICS_SCHEMA_ID,
    MATCH_PROFILE_DISPLAY_SRGB,
    DiagnosticsV1,
    PromotionDecision,
    ReferenceMatchContractError,
    adjudicate_core_acceptance,
    core_acceptance_from_json,
    core_acceptance_to_json,
    make_capabilities,
    make_match_view,
    make_transform_bundle,
    validate_core_acceptance_decision,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_core_acceptance_v1.schema.json"
)
SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
SHA_D = "4" * 64
CONTRACT_ID = "zhuise.transform-bundle.v1"
ALGORITHM_ID = "zhuise.dpct.v1"


def _objects(status: str = "ok"):
    source = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=SHA_A,
        shape=(2, 3, 3),
        render_bridge_id="neuro-film.working-image-direct.v1",
        provenance_fingerprint=SHA_B,
    )
    reference = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=SHA_C,
        shape=(2, 3, 3),
        render_bridge_id="neuro-film.working-image-direct.v1",
        provenance_fingerprint=SHA_D,
    )
    transform = make_transform_bundle(
        producer_id="zhuise-core",
        producer_version="0.1.0",
        producer_build_sha256=SHA_D,
        contract_schema_id=CONTRACT_ID,
        algorithm_id=ALGORITHM_ID,
        algorithm_version="1.0.0",
        intent_id=SHA_C,
        source_view_id=source.view_id,
        reference_view_id=reference.view_id,
        payload_schema_id="zhuise.dpct-transform.v1",
        payload_sha256=SHA_A,
        capability_requirements=["source-bound-fit"],
    )
    capabilities = make_capabilities(
        producer_id="zhuise-core",
        producer_version="0.1.0",
        producer_build_sha256=SHA_D,
        contract_schema_ids=[CONTRACT_ID],
        supported_profile_ids=[MATCH_PROFILE_DISPLAY_SRGB],
        supported_algorithm_ids=[ALGORITHM_ID],
        feature_flags=["source-bound-fit"],
    )
    diagnostics = DiagnosticsV1(
        schema_id=DIAGNOSTICS_SCHEMA_ID,
        transform_id=transform.transform_id,
        status=status,
        fallback_reason=None if status == "ok" else "unsupported-rail",
        finite=True,
        out_of_gamut_fraction=0.01,
        clipping_fraction=0.0,
        projected_fraction=0.01,
        confidence=0.75,
        timing_ms=2.0,
        backend_id="cpu-reference",
        backend_fingerprint=SHA_B,
        warnings=(),
    )
    return source, reference, transform, capabilities, diagnostics


def _adjudicate(
    promotion: PromotionDecision,
    *,
    status: str = "ok",
    override: bool = False,
):
    source, reference, transform, capabilities, diagnostics = _objects(
        status
    )
    return adjudicate_core_acceptance(
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
        promotion=promotion,
        allow_research_baseline=override,
    )


def test_promoted_ok_core_only_enters_product_guard() -> None:
    decision = _adjudicate(PromotionDecision("promoted", ()))
    assert decision.schema_id == CORE_ACCEPTANCE_SCHEMA_ID
    assert decision.accepted_for_product_guard
    assert decision.action == "candidate-for-product-guard"
    assert decision.reasons == ()
    assert decision.required_gates == ("A1", "A4", "A5")


def test_unpromoted_core_defaults_to_identity() -> None:
    decision = _adjudicate(
        PromotionDecision("rejected", ("known-operator-tail",))
    )
    assert not decision.accepted_for_product_guard
    assert decision.action == "identity-fallback"
    assert decision.reasons == ("algorithm-not-promoted",)
    assert decision.promotion_reasons == ("known-operator-tail",)


def test_research_override_still_only_enters_product_guard() -> None:
    decision = _adjudicate(
        PromotionDecision("rejected", ("known-operator-tail",)),
        override=True,
    )
    assert decision.accepted_for_product_guard
    assert decision.action == "candidate-for-product-guard"
    assert decision.research_baseline_override
    assert decision.reasons == ()


@pytest.mark.parametrize(
    "status",
    ["identity-fallback", "unsupported", "invalid"],
)
def test_non_ok_core_fails_closed_even_with_research_override(
    status: str,
) -> None:
    decision = _adjudicate(
        PromotionDecision("rejected", ("known-operator-tail",)),
        status=status,
        override=True,
    )
    assert not decision.accepted_for_product_guard
    assert decision.action == "identity-fallback"
    assert f"core-status:{status}" in decision.reasons
    assert "core-fallback:unsupported-rail" in decision.reasons


def test_decision_roundtrip_schema_and_identity() -> None:
    decision = _adjudicate(PromotionDecision("promoted", ()))
    encoded = core_acceptance_to_json(decision)
    assert core_acceptance_from_json(encoded) == decision
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))
    assert (
        core_acceptance_from_json(core_acceptance_to_json(decision)).decision_id
        == decision.decision_id
    )


def test_parser_rejects_unknown_fields() -> None:
    payload = json.loads(
        core_acceptance_to_json(
            _adjudicate(PromotionDecision("promoted", ()))
        )
    )
    payload["surprise"] = True
    with pytest.raises(ReferenceMatchContractError, match="keys mismatch"):
        core_acceptance_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {"action": "identity-fallback"},
            "action is inconsistent",
        ),
        (
            {"required_gates": ("A1", "A4")},
            "bind A1/A4/A5",
        ),
        (
            {"accepted_for_product_guard": False},
            "state is inconsistent",
        ),
    ],
)
def test_decision_validation_rejects_inconsistent_state(
    mutation,
    message,
) -> None:
    decision = _adjudicate(PromotionDecision("promoted", ()))
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_core_acceptance_decision(replace(decision, **mutation))


def test_promotion_decision_state_is_validated() -> None:
    with pytest.raises(
        ReferenceMatchContractError,
        match="promoted decision",
    ):
        _adjudicate(PromotionDecision("promoted", ("surprise",)))
    with pytest.raises(
        ReferenceMatchContractError,
        match="unpromoted decision",
    ):
        _adjudicate(PromotionDecision("rejected", ()))
