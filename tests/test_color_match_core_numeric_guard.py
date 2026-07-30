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
    CORE_NUMERIC_GUARD_CLAIM_CEILING,
    MATCH_PROFILE_DISPLAY_SRGB,
    CoreNumericGuardPolicyV1,
    PreparedMatchViewV1,
    PromotionDecision,
    ReferenceMatchContractError,
    adapt_dpct_candidate_v2,
    admit_core_apply_receipt,
    adjudicate_core_acceptance,
    core_numeric_guard_decision_from_json,
    core_numeric_guard_decision_to_json,
    core_numeric_guard_policy_id,
    guard_core_candidate_numeric_v1,
    make_match_view,
    validate_core_numeric_guard_decision_v1,
    validate_core_numeric_guard_policy_v1,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "zhuise_producer_contract_exact_bits_v2.json"
)
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_core_numeric_guard_decision_v1.schema.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _prepared(
    encoded_hex: str,
    provenance: str,
) -> PreparedMatchViewV1:
    wire = bytes.fromhex(encoded_hex)
    pixels = np.frombuffer(wire, dtype=">f4").astype(np.float32, copy=True)
    pixels = np.ascontiguousarray(pixels.reshape(2, 2, 3))
    pixels.flags.writeable = False
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=(2, 2, 3),
        render_bridge_id="neuro-film.numeric-guard-test.v1",
        provenance_fingerprint=provenance * 64,
    )
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _execution(*, promoted: bool = True):
    fixture = _fixture()
    source = _prepared(fixture["source_pixel_f32be_hex"], "1")
    reference = _prepared(fixture["reference_pixel_f32be_hex"], "2")
    candidate = adapt_dpct_candidate_v2(
        source=source,
        reference=reference,
        producer_source=fixture["source"],
        producer_reference=fixture["reference"],
        producer_transform=fixture["transform"],
        producer_transform_payload=bytes.fromhex(fixture["payload_hex"]),
        producer_diagnostics=fixture["diagnostics"],
        producer_apply_result=fixture["apply_result"],
        output_pixel_f32be=bytes.fromhex(
            fixture["output_pixel_f32be_hex"]
        ),
        intent_id="3" * 64,
    )
    promotion = (
        PromotionDecision("promoted", ())
        if promoted
        else PromotionDecision("rejected", ("not-promoted",))
    )
    acceptance = adjudicate_core_acceptance(
        source=source.descriptor,
        reference=reference.descriptor,
        transform=candidate.transform,
        capabilities=candidate.capabilities,
        diagnostics=candidate.diagnostics,
        promotion=promotion,
    )
    admission = admit_core_apply_receipt(
        prepared=candidate.prepared_output,
        acceptance=acceptance,
        source=source.descriptor,
        reference=reference.descriptor,
        transform=candidate.transform,
        capabilities=candidate.capabilities,
        diagnostics=candidate.diagnostics,
    )
    return source, reference, candidate, acceptance, admission


def _relaxed() -> CoreNumericGuardPolicyV1:
    return CoreNumericGuardPolicyV1(
        max_out_of_gamut_fraction=1.0,
        max_clipping_fraction=1.0,
        max_projected_fraction=1.0,
        max_new_boundary_fraction=1.0,
    )


def _make_reason_inconsistent(value):
    policy = replace(value.policy, max_clipping_fraction=0.5)
    return replace(
        value,
        policy=policy,
        policy_id=core_numeric_guard_policy_id(policy),
        clipping_fraction=0.9,
    )


def test_relaxed_numeric_guard_is_only_transaction_eligible() -> None:
    source, reference, candidate, acceptance, admission = _execution()
    decision = guard_core_candidate_numeric_v1(
        source=source,
        reference=reference,
        candidate=candidate,
        acceptance=acceptance,
        admission=admission,
        policy=_relaxed(),
    )
    assert decision.accepted_for_transaction
    assert decision.action == "eligible-for-transaction"
    assert decision.reasons == ()
    assert decision.claim_ceiling == CORE_NUMERIC_GUARD_CLAIM_CEILING
    assert decision.clipping_fraction == 0.25
    assert decision.new_boundary_fraction == 0.0
    assert "applied" not in core_numeric_guard_decision_to_json(decision)


def test_default_policy_rejects_fixture_clipping() -> None:
    source, reference, candidate, acceptance, admission = _execution()
    decision = guard_core_candidate_numeric_v1(
        source=source,
        reference=reference,
        candidate=candidate,
        acceptance=acceptance,
        admission=admission,
    )
    assert not decision.accepted_for_transaction
    assert decision.action == "identity-fallback"
    assert decision.reasons == ("clipping-fraction",)


def test_threshold_equality_passes() -> None:
    source, reference, candidate, acceptance, admission = _execution()
    policy = CoreNumericGuardPolicyV1(
        max_out_of_gamut_fraction=0.25,
        max_clipping_fraction=0.25,
        max_projected_fraction=0.0,
        max_new_boundary_fraction=0.0,
    )
    decision = guard_core_candidate_numeric_v1(
        source=source,
        reference=reference,
        candidate=candidate,
        acceptance=acceptance,
        admission=admission,
        policy=policy,
    )
    assert decision.accepted_for_transaction


def test_rejected_admission_cannot_pass_relaxed_numeric_policy() -> None:
    source, reference, candidate, acceptance, admission = _execution(
        promoted=False
    )
    decision = guard_core_candidate_numeric_v1(
        source=source,
        reference=reference,
        candidate=candidate,
        acceptance=acceptance,
        admission=admission,
        policy=_relaxed(),
    )
    assert decision.reasons == ("admission-identity-fallback",)
    assert not decision.admission_accepted_for_product_guard


def test_decision_roundtrip_and_schema_are_strict() -> None:
    source, reference, candidate, acceptance, admission = _execution()
    decision = guard_core_candidate_numeric_v1(
        source=source,
        reference=reference,
        candidate=candidate,
        acceptance=acceptance,
        admission=admission,
        policy=_relaxed(),
    )
    encoded = core_numeric_guard_decision_to_json(decision)
    assert core_numeric_guard_decision_from_json(encoded) == decision
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))


@pytest.mark.parametrize(
    "policy",
    [
        CoreNumericGuardPolicyV1(max_clipping_fraction=-0.1),
        CoreNumericGuardPolicyV1(max_projected_fraction=float("nan")),
        CoreNumericGuardPolicyV1(boundary_epsilon=0.5),
        CoreNumericGuardPolicyV1(
            schema_id="neuro-film.core-numeric-guard-policy.v2"
        ),
    ],
)
def test_invalid_policies_fail_closed(policy) -> None:
    with pytest.raises(ReferenceMatchContractError):
        validate_core_numeric_guard_policy_v1(policy)


def test_mutated_output_and_swapped_source_fail_before_metrics() -> None:
    source, reference, candidate, acceptance, admission = _execution()
    candidate.prepared_output.pixels.flags.writeable = True
    candidate.prepared_output.pixels[0, 0, 0] += np.float32(0.25)
    candidate.prepared_output.pixels.flags.writeable = False
    with pytest.raises(
        ReferenceMatchContractError,
        match="pixel identity",
    ):
        guard_core_candidate_numeric_v1(
            source=source,
            reference=reference,
            candidate=candidate,
            acceptance=acceptance,
            admission=admission,
            policy=_relaxed(),
        )

    source, reference, candidate, acceptance, admission = _execution()
    swapped = PreparedMatchViewV1(
        descriptor=replace(
            source.descriptor,
            provenance_fingerprint="4" * 64,
        ),
        pixels=source.pixels,
    )
    with pytest.raises(ReferenceMatchContractError):
        guard_core_candidate_numeric_v1(
            source=swapped,
            reference=reference,
            candidate=candidate,
            acceptance=acceptance,
            admission=admission,
            policy=_relaxed(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, action="applied"),
            "action is unsupported",
        ),
        (
            lambda value: replace(
                value,
                accepted_for_transaction=False,
            ),
            "state is inconsistent",
        ),
        (
            _make_reason_inconsistent,
            "reasons are inconsistent",
        ),
        (
            lambda value: replace(value, policy_id="0" * 64),
            "policy_id mismatch",
        ),
        (
            lambda value: replace(
                value,
                claim_ceiling="visual-safe",
            ),
            "claim ceiling mismatch",
        ),
    ],
)
def test_decision_mutations_fail_closed(mutation, message) -> None:
    source, reference, candidate, acceptance, admission = _execution()
    decision = guard_core_candidate_numeric_v1(
        source=source,
        reference=reference,
        candidate=candidate,
        acceptance=acceptance,
        admission=admission,
        policy=_relaxed(),
    )
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_core_numeric_guard_decision_v1(mutation(decision))


def test_unknown_json_field_fails_closed() -> None:
    source, reference, candidate, acceptance, admission = _execution()
    decision = guard_core_candidate_numeric_v1(
        source=source,
        reference=reference,
        candidate=candidate,
        acceptance=acceptance,
        admission=admission,
        policy=_relaxed(),
    )
    payload = deepcopy(decision.to_dict())
    payload["surprise"] = True
    with pytest.raises(ReferenceMatchContractError, match="keys mismatch"):
        core_numeric_guard_decision_from_json(json.dumps(payload))
