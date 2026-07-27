from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from src.color_match import (
    CORE_CANDIDATE_ADMISSION_SCHEMA_ID,
    DIAGNOSTICS_SCHEMA_ID,
    MATCH_PROFILE_DISPLAY_SRGB,
    DiagnosticsV1,
    PromotionDecision,
    ReferenceMatchContractError,
    admit_core_apply_receipt,
    adjudicate_core_acceptance,
    core_candidate_admission_from_json,
    core_candidate_admission_to_json,
    make_capabilities,
    make_match_view,
    make_transform_bundle,
    prepare_core_apply_receipt,
    validate_core_candidate_admission,
    validate_core_candidate_admission_binding,
    validate_prepared_core_apply_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_core_candidate_admission_v2.schema.json"
)
SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
SHA_D = "4" * 64
CONTRACT_ID = "fixture.transform-bundle.v1"
ALGORITHM_ID = "fixture.algorithm.v1"


def _execution(
    promotion: PromotionDecision,
    *,
    override: bool = False,
    seed: float = 0.0,
):
    source = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=SHA_A,
        shape=(2, 2, 3),
        render_bridge_id="neuro-film.working-image-direct.v1",
        provenance_fingerprint=SHA_B,
    )
    reference = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=SHA_C,
        shape=(2, 2, 3),
        render_bridge_id="neuro-film.working-image-direct.v1",
        provenance_fingerprint=SHA_D,
    )
    transform = make_transform_bundle(
        producer_id="fixture-core",
        producer_version="1.0.0",
        producer_build_sha256=SHA_D,
        contract_schema_id=CONTRACT_ID,
        algorithm_id=ALGORITHM_ID,
        algorithm_version="1.0.0",
        intent_id=SHA_C,
        source_view_id=source.view_id,
        reference_view_id=reference.view_id,
        payload_schema_id="fixture.payload.v1",
        payload_sha256=SHA_A,
        capability_requirements=["source-bound-fit"],
    )
    capabilities = make_capabilities(
        producer_id="fixture-core",
        producer_version="1.0.0",
        producer_build_sha256=SHA_D,
        contract_schema_ids=[CONTRACT_ID],
        supported_profile_ids=[MATCH_PROFILE_DISPLAY_SRGB],
        supported_algorithm_ids=[ALGORITHM_ID],
        feature_flags=["source-bound-fit"],
    )
    diagnostics = DiagnosticsV1(
        schema_id=DIAGNOSTICS_SCHEMA_ID,
        transform_id=transform.transform_id,
        status="ok",
        fallback_reason=None,
        finite=True,
        out_of_gamut_fraction=0.01,
        clipping_fraction=0.0,
        projected_fraction=0.01,
        confidence=0.75,
        timing_ms=2.0,
        backend_id="fixture-cpu",
        backend_fingerprint=SHA_B,
        warnings=(),
    )
    pixels = (
        np.linspace(-0.1, 1.2, 12, dtype=np.float32).reshape(2, 2, 3)
        + np.float32(seed)
    )
    prepared = prepare_core_apply_receipt(
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
        output_pixels=pixels,
    )
    acceptance = adjudicate_core_acceptance(
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
        promotion=promotion,
        allow_research_baseline=override,
    )
    admission = admit_core_apply_receipt(
        prepared=prepared,
        acceptance=acceptance,
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
    )
    return (
        admission,
        prepared,
        acceptance,
        source,
        reference,
        transform,
        capabilities,
        diagnostics,
    )


def test_promoted_receipt_is_only_pending_product_guard() -> None:
    admission, prepared, acceptance, *_ = _execution(
        PromotionDecision("promoted", ())
    )
    assert admission.schema_id == CORE_CANDIDATE_ADMISSION_SCHEMA_ID
    assert admission.accepted_for_product_guard
    assert admission.action == "candidate-for-product-guard"
    assert admission.guard_state == "pending-product-guard"
    assert admission.apply_receipt_id == prepared.receipt.receipt_id
    assert admission.acceptance_decision_id == acceptance.decision_id
    assert admission.required_gates == ("A1", "A4", "A5")


def test_unpromoted_receipt_remains_identity_fallback() -> None:
    admission, *_ = _execution(
        PromotionDecision("rejected", ("quality-gate",))
    )
    assert not admission.accepted_for_product_guard
    assert admission.action == "identity-fallback"
    assert admission.guard_state == "identity-fallback"


def test_research_override_still_waits_for_product_guard() -> None:
    admission, *_ = _execution(
        PromotionDecision("rejected", ("quality-gate",)),
        override=True,
    )
    assert admission.accepted_for_product_guard
    assert admission.guard_state == "pending-product-guard"


def test_admission_roundtrip_and_schema_are_strict() -> None:
    admission, *_ = _execution(PromotionDecision("promoted", ()))
    encoded = core_candidate_admission_to_json(admission)
    assert core_candidate_admission_from_json(encoded) == admission
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))


def test_mutated_pixels_cannot_be_admitted() -> None:
    (
        _,
        prepared,
        acceptance,
        source,
        reference,
        transform,
        capabilities,
        diagnostics,
    ) = _execution(PromotionDecision("promoted", ()))
    prepared.pixels.flags.writeable = True
    prepared.pixels[0, 0, 0] += np.float32(0.5)
    prepared.pixels.flags.writeable = False
    with pytest.raises(
        ReferenceMatchContractError,
        match="pixel identity",
    ):
        admit_core_apply_receipt(
            prepared=prepared,
            acceptance=acceptance,
            source=source,
            reference=reference,
            transform=transform,
            capabilities=capabilities,
            diagnostics=diagnostics,
        )


def test_other_receipt_cannot_replace_admitted_pixels() -> None:
    first = _execution(PromotionDecision("promoted", ()), seed=0.0)
    second = _execution(PromotionDecision("promoted", ()), seed=0.25)
    with pytest.raises(
        ReferenceMatchContractError,
        match="binding mismatch",
    ):
        validate_core_candidate_admission_binding(
            first[0],
            receipt=second[1].receipt,
            acceptance=first[2],
        )


def test_other_acceptance_cannot_replace_bound_decision() -> None:
    promoted = _execution(PromotionDecision("promoted", ()))
    rejected = _execution(
        PromotionDecision("rejected", ("quality-gate",))
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="binding mismatch",
    ):
        validate_core_candidate_admission_binding(
            promoted[0],
            receipt=promoted[1].receipt,
            acceptance=rejected[2],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {"guard_state": "applied"},
            "unsupported core candidate guard state",
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
def test_admission_state_mutations_fail_closed(mutation, message) -> None:
    admission, *_ = _execution(PromotionDecision("promoted", ()))
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_core_candidate_admission(
            replace(admission, **mutation)
        )


def test_parser_rejects_unknown_fields() -> None:
    admission, *_ = _execution(PromotionDecision("promoted", ()))
    payload = admission.to_dict()
    payload["surprise"] = True
    with pytest.raises(ReferenceMatchContractError, match="keys mismatch"):
        core_candidate_admission_from_json(json.dumps(payload))
