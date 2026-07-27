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
    CORE_NUMERIC_BATCH_GUARD_SCHEMA_ID,
    MATCH_PROFILE_DISPLAY_SRGB,
    CoreNumericGuardPolicyV1,
    PreparedMatchViewV1,
    PromotionDecision,
    ReferenceMatchContractError,
    adapt_dpct_candidate_v2,
    admit_core_apply_receipt,
    adjudicate_core_acceptance,
    core_numeric_batch_guard_from_json,
    core_numeric_batch_guard_to_json,
    guard_core_candidate_numeric_v1,
    guard_core_numeric_batch_v1,
    make_match_view,
    resolve_dpct_batch_v1,
    validate_core_numeric_batch_guard_v1,
)
from src.color_match.canonical import canonical_sha256


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
    / "reference_core_numeric_batch_guard_v1.schema.json"
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
        render_bridge_id="neuro-film.numeric-batch-test.v1",
        provenance_fingerprint=provenance * 64,
    )
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _candidate(
    reference: PreparedMatchViewV1,
    *,
    provenance: str,
    intent: str,
    promoted: bool = True,
):
    fixture = _fixture()
    source = _prepared(fixture["source_pixel_f32be_hex"], provenance)
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
        intent_id=intent * 64,
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
    return source, candidate, acceptance, admission


def _pending_batch():
    fixture = _fixture()
    reference = _prepared(fixture["reference_pixel_f32be_hex"], "a")
    first = _candidate(reference, provenance="b", intent="1")
    second = _candidate(reference, provenance="c", intent="2")
    batch = resolve_dpct_batch_v1(
        reference=reference,
        sources=(first[0], second[0]),
        outcomes=(first[1], second[1]),
        adjudications=(
            (first[2], first[3]),
            (second[2], second[3]),
        ),
    )
    return reference, (first, second), batch


def _decision(reference, item, policy):
    return guard_core_candidate_numeric_v1(
        source=item[0],
        reference=reference,
        candidate=item[1],
        acceptance=item[2],
        admission=item[3],
        policy=policy,
    )


def _relaxed() -> CoreNumericGuardPolicyV1:
    return CoreNumericGuardPolicyV1(
        max_out_of_gamut_fraction=1.0,
        max_clipping_fraction=1.0,
        max_projected_fraction=1.0,
        max_new_boundary_fraction=1.0,
    )


def _with_clipping_failure(decision):
    provisional = replace(
        decision,
        decision_id="0" * 64,
        accepted_for_transaction=False,
        action="identity-fallback",
        reasons=("clipping-fraction",),
        clipping_fraction=0.9,
    )
    payload = provisional.to_dict()
    payload.pop("decision_id")
    return replace(
        provisional,
        decision_id=canonical_sha256(payload),
    )


def test_all_numeric_passes_only_make_batch_transaction_eligible() -> None:
    reference, items, batch = _pending_batch()
    decisions = tuple(
        _decision(reference, item, _relaxed()) for item in items
    )
    result = guard_core_numeric_batch_v1(
        batch=batch,
        decisions=decisions,
    )
    assert result.schema_id == CORE_NUMERIC_BATCH_GUARD_SCHEMA_ID
    assert result.atomic_state == "eligible-for-transaction"
    assert result.numeric_policy_id == decisions[0].policy_id
    assert all(
        row.individual_action == "eligible-for-transaction"
        for row in result.sources
    )
    encoded = core_numeric_batch_guard_to_json(result)
    assert "applied" not in encoded
    assert "pixels" not in encoded


def test_one_numeric_failure_forces_whole_batch_fallback() -> None:
    reference, items, batch = _pending_batch()
    policy = CoreNumericGuardPolicyV1(
        max_out_of_gamut_fraction=1.0,
        max_clipping_fraction=0.5,
        max_projected_fraction=1.0,
        max_new_boundary_fraction=1.0,
    )
    first = _decision(reference, items[0], policy)
    second = _with_clipping_failure(
        _decision(reference, items[1], policy)
    )
    result = guard_core_numeric_batch_v1(
        batch=batch,
        decisions=(first, second),
    )
    assert result.atomic_state == "identity-fallback"
    assert [row.individual_action for row in result.sources] == [
        "eligible-for-transaction",
        "identity-fallback",
    ]


def test_upstream_fallback_short_circuits_numeric_evaluation() -> None:
    fixture = _fixture()
    reference = _prepared(fixture["reference_pixel_f32be_hex"], "a")
    item = _candidate(
        reference,
        provenance="b",
        intent="1",
        promoted=False,
    )
    batch = resolve_dpct_batch_v1(
        reference=reference,
        sources=(item[0],),
        outcomes=(item[1],),
        adjudications=((item[2], item[3]),),
    )
    result = guard_core_numeric_batch_v1(batch=batch)
    assert result.atomic_state == "identity-fallback"
    assert result.numeric_policy_id is None
    assert result.sources[0].individual_action == "not-evaluated"
    assert result.sources[0].numeric_decision_id is None
    with pytest.raises(
        ReferenceMatchContractError,
        match="cannot accept numeric decisions",
    ):
        guard_core_numeric_batch_v1(
            batch=batch,
            decisions=(
                _decision(reference, item, _relaxed()),
            ),
        )


def test_decision_cannot_be_reordered_or_reassigned() -> None:
    reference, items, batch = _pending_batch()
    decisions = tuple(
        _decision(reference, item, _relaxed()) for item in items
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="source binding mismatch",
    ):
        guard_core_numeric_batch_v1(
            batch=batch,
            decisions=tuple(reversed(decisions)),
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="one numeric decision per source",
    ):
        guard_core_numeric_batch_v1(
            batch=batch,
            decisions=decisions[:1],
        )


def test_batch_roundtrip_schema_and_repeat_are_exact() -> None:
    reference, items, batch = _pending_batch()
    decisions = tuple(
        _decision(reference, item, _relaxed()) for item in items
    )
    one = guard_core_numeric_batch_v1(
        batch=batch,
        decisions=decisions,
    )
    two = guard_core_numeric_batch_v1(
        batch=batch,
        decisions=decisions,
    )
    assert one == two
    encoded = core_numeric_batch_guard_to_json(one)
    assert core_numeric_batch_guard_from_json(encoded) == one
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, atomic_state="applied"),
            "atomic state is unsupported",
        ),
        (
            lambda value: replace(value, numeric_policy_id=None),
            "must bind one policy",
        ),
        (
            lambda value: replace(
                value,
                sources=(
                    replace(value.sources[0], source_index=1),
                    value.sources[1],
                ),
            ),
            "indices must be contiguous",
        ),
        (
            lambda value: replace(
                value,
                sources=(
                    replace(
                        value.sources[0],
                        numeric_accepted_for_transaction=False,
                    ),
                    value.sources[1],
                ),
            ),
            "source decision state is inconsistent",
        ),
        (
            lambda value: replace(value, claim_ceiling="visual-safe"),
            "claim ceiling mismatch",
        ),
    ],
)
def test_batch_mutations_fail_closed(mutation, message) -> None:
    reference, items, batch = _pending_batch()
    decisions = tuple(
        _decision(reference, item, _relaxed()) for item in items
    )
    result = guard_core_numeric_batch_v1(
        batch=batch,
        decisions=decisions,
    )
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_core_numeric_batch_guard_v1(mutation(result))


def test_unknown_json_field_fails_closed() -> None:
    reference, items, batch = _pending_batch()
    decisions = tuple(
        _decision(reference, item, _relaxed()) for item in items
    )
    result = guard_core_numeric_batch_v1(
        batch=batch,
        decisions=decisions,
    )
    payload = deepcopy(result.to_dict())
    payload["surprise"] = True
    with pytest.raises(ReferenceMatchContractError, match="keys mismatch"):
        core_numeric_batch_guard_from_json(json.dumps(payload))
