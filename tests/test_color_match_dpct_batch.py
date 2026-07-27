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
    DPCT_BATCH_RESOLUTION_SCHEMA_ID,
    MATCH_PROFILE_DISPLAY_SRGB,
    PreparedMatchViewV1,
    PromotionDecision,
    ReferenceMatchContractError,
    adapt_dpct_candidate_v2,
    admit_core_apply_receipt,
    adjudicate_core_acceptance,
    dpct_batch_resolution_from_json,
    dpct_batch_resolution_to_json,
    make_match_view,
    resolve_dpct_batch_v1,
    validate_dpct_batch_resolution_v1,
    verify_dpct_failed_diagnostics_v2,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "zhuise_producer_contract_exact_bits_v2.json"
)
FAILED_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "zhuise_producer_contract_failed_exact_v2.json"
)
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_dpct_batch_resolution_v1.schema.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepared(encoded_hex: str, provenance: str) -> PreparedMatchViewV1:
    wire = bytes.fromhex(encoded_hex)
    pixels = np.frombuffer(wire, dtype=">f4").astype(np.float32, copy=True)
    pixels = np.ascontiguousarray(pixels.reshape(2, 2, 3))
    pixels.flags.writeable = False
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=(2, 2, 3),
        render_bridge_id="neuro-film.dpct-batch-test.v1",
        provenance_fingerprint=provenance * 64,
    )
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _candidate(
    reference: PreparedMatchViewV1,
    *,
    provenance: str,
    intent: str,
    promoted: bool,
):
    fixture = _json(FIXTURE)
    source = _prepared(
        fixture["source_pixel_f32be_hex"],
        provenance,
    )
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
    return source, candidate, (acceptance, admission)


def _reference() -> PreparedMatchViewV1:
    fixture = _json(FIXTURE)
    return _prepared(fixture["reference_pixel_f32be_hex"], "a")


def test_all_promoted_sources_stop_at_pending_product_guard() -> None:
    reference = _reference()
    first = _candidate(
        reference, provenance="b", intent="1", promoted=True
    )
    second = _candidate(
        reference, provenance="c", intent="2", promoted=True
    )
    result = resolve_dpct_batch_v1(
        reference=reference,
        sources=(first[0], second[0]),
        outcomes=(first[1], second[1]),
        adjudications=(first[2], second[2]),
    )
    assert result.schema_id == DPCT_BATCH_RESOLUTION_SCHEMA_ID
    assert result.atomic_state == "pending-product-guard"
    assert result.required_gates == ("A1", "A4", "A5")
    assert [row.source_index for row in result.sources] == [0, 1]
    assert {
        row.individual_action for row in result.sources
    } == {"candidate-for-product-guard"}
    assert all(row.apply_receipt_id for row in result.sources)
    assert "applied" not in dpct_batch_resolution_to_json(result)


def test_one_admission_fallback_forces_whole_batch_fallback() -> None:
    reference = _reference()
    first = _candidate(
        reference, provenance="b", intent="1", promoted=True
    )
    second = _candidate(
        reference, provenance="c", intent="2", promoted=False
    )
    result = resolve_dpct_batch_v1(
        reference=reference,
        sources=(first[0], second[0]),
        outcomes=(first[1], second[1]),
        adjudications=(first[2], second[2]),
    )
    assert result.atomic_state == "identity-fallback"
    assert [row.individual_action for row in result.sources] == [
        "candidate-for-product-guard",
        "identity-fallback",
    ]


def test_producer_failure_short_circuits_all_admission() -> None:
    reference = _reference()
    candidate = _candidate(
        reference, provenance="b", intent="1", promoted=True
    )
    failure = verify_dpct_failed_diagnostics_v2(
        _json(FAILED_FIXTURE)["diagnostics"]
    )
    failed_source = _prepared(
        _json(FIXTURE)["source_pixel_f32be_hex"],
        "c",
    )
    result = resolve_dpct_batch_v1(
        reference=reference,
        sources=(candidate[0], failed_source),
        outcomes=(candidate[1], failure),
    )
    assert result.atomic_state == "identity-fallback"
    assert result.sources[0].individual_action == "not-evaluated"
    assert result.sources[0].acceptance_decision_id is None
    assert result.sources[1].individual_action == "identity-fallback"
    assert result.sources[1].consumer_transform_id is None
    assert result.sources[1].apply_receipt_id is None


def test_failed_batch_rejects_partial_adjudication() -> None:
    reference = _reference()
    candidate = _candidate(
        reference, provenance="b", intent="1", promoted=True
    )
    failure = verify_dpct_failed_diagnostics_v2(
        _json(FAILED_FIXTURE)["diagnostics"]
    )
    failed_source = _prepared(
        _json(FIXTURE)["source_pixel_f32be_hex"],
        "c",
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="short-circuit before admission",
    ):
        resolve_dpct_batch_v1(
            reference=reference,
            sources=(candidate[0], failed_source),
            outcomes=(candidate[1], failure),
            adjudications=(candidate[2], candidate[2]),
        )


def test_batch_roundtrip_schema_and_repeat_are_exact() -> None:
    reference = _reference()
    first = _candidate(
        reference, provenance="b", intent="1", promoted=True
    )
    second = _candidate(
        reference, provenance="c", intent="2", promoted=True
    )
    kwargs = {
        "reference": reference,
        "sources": (first[0], second[0]),
        "outcomes": (first[1], second[1]),
        "adjudications": (first[2], second[2]),
    }
    one = resolve_dpct_batch_v1(**kwargs)
    two = resolve_dpct_batch_v1(**kwargs)
    assert one == two
    encoded = dpct_batch_resolution_to_json(one)
    assert dpct_batch_resolution_from_json(encoded) == one
    schema = _json(SCHEMA)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))


def test_source_order_is_explicit_and_changes_batch_identity() -> None:
    reference = _reference()
    first = _candidate(
        reference, provenance="b", intent="1", promoted=True
    )
    second = _candidate(
        reference, provenance="c", intent="2", promoted=True
    )
    forward = resolve_dpct_batch_v1(
        reference=reference,
        sources=(first[0], second[0]),
        outcomes=(first[1], second[1]),
        adjudications=(first[2], second[2]),
    )
    reverse = resolve_dpct_batch_v1(
        reference=reference,
        sources=(second[0], first[0]),
        outcomes=(second[1], first[1]),
        adjudications=(second[2], first[2]),
    )
    assert forward.batch_id != reverse.batch_id
    assert (
        forward.sources[0].source_view_id
        == reverse.sources[1].source_view_id
    )


def test_candidate_cannot_be_reassigned_to_other_source() -> None:
    reference = _reference()
    first = _candidate(
        reference, provenance="b", intent="1", promoted=True
    )
    second = _candidate(
        reference, provenance="c", intent="2", promoted=True
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="consumer view binding mismatch",
    ):
        resolve_dpct_batch_v1(
            reference=reference,
            sources=(first[0], second[0]),
            outcomes=(second[1], first[1]),
            adjudications=(second[2], first[2]),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, atomic_state="applied"),
            "atomic_state is unsupported",
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
            lambda value: replace(value, required_gates=("A1", "A4")),
            "bind A1/A4/A5",
        ),
        (
            lambda value: replace(
                value,
                atomic_state="identity-fallback",
            ),
            "atomic state is inconsistent",
        ),
    ],
)
def test_batch_mutations_fail_closed(mutation, message) -> None:
    reference = _reference()
    first = _candidate(
        reference, provenance="b", intent="1", promoted=True
    )
    second = _candidate(
        reference, provenance="c", intent="2", promoted=True
    )
    result = resolve_dpct_batch_v1(
        reference=reference,
        sources=(first[0], second[0]),
        outcomes=(first[1], second[1]),
        adjudications=(first[2], second[2]),
    )
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_dpct_batch_resolution_v1(mutation(result))


def test_unknown_json_field_and_empty_batch_fail_closed() -> None:
    reference = _reference()
    with pytest.raises(
        ReferenceMatchContractError,
        match="must not be empty",
    ):
        resolve_dpct_batch_v1(
            reference=reference,
            sources=(),
            outcomes=(),
        )
    first = _candidate(
        reference, provenance="b", intent="1", promoted=True
    )
    result = resolve_dpct_batch_v1(
        reference=reference,
        sources=(first[0],),
        outcomes=(first[1],),
        adjudications=(first[2],),
    )
    payload = deepcopy(result.to_dict())
    payload["surprise"] = True
    with pytest.raises(ReferenceMatchContractError, match="keys mismatch"):
        dpct_batch_resolution_from_json(json.dumps(payload))
