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
    CORE_PRODUCT_AUTHORIZATION_SCHEMA_ID,
    MATCH_PROFILE_DISPLAY_SRGB,
    CoreNumericGuardPolicyV1,
    PreparedMatchViewV1,
    PromotionDecision,
    ReferenceMatchContractError,
    adapt_dpct_candidate_v2,
    admit_core_apply_receipt,
    adjudicate_core_acceptance,
    authorize_core_product_staging_v1,
    core_product_authorization_from_json,
    core_product_authorization_to_json,
    guard_core_candidate_numeric_v1,
    guard_core_numeric_batch_v1,
    make_match_view,
    resolve_dpct_batch_v1,
    validate_core_product_staging_authorization_v1,
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
    / "reference_core_product_staging_authorization_v1.schema.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _prepared(encoded_hex: str, provenance: str) -> PreparedMatchViewV1:
    wire = bytes.fromhex(encoded_hex)
    pixels = np.frombuffer(wire, dtype=">f4").astype(np.float32, copy=True)
    pixels = np.ascontiguousarray(pixels.reshape(2, 2, 3))
    pixels.flags.writeable = False
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=(2, 2, 3),
        render_bridge_id="neuro-film.product-auth-test.v1",
        provenance_fingerprint=provenance * 64,
    )
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _candidate(
    reference: PreparedMatchViewV1,
    *,
    provenance: str,
    intent: str,
    promotion: PromotionDecision,
    research_override: bool,
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
    acceptance = adjudicate_core_acceptance(
        source=source.descriptor,
        reference=reference.descriptor,
        transform=candidate.transform,
        capabilities=candidate.capabilities,
        diagnostics=candidate.diagnostics,
        promotion=promotion,
        allow_research_baseline=research_override,
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


def _pipeline(*, second_research_override: bool = False):
    fixture = _fixture()
    reference = _prepared(fixture["reference_pixel_f32be_hex"], "a")
    first = _candidate(
        reference,
        provenance="b",
        intent="1",
        promotion=PromotionDecision("promoted", ()),
        research_override=False,
    )
    second = _candidate(
        reference,
        provenance="c",
        intent="2",
        promotion=(
            PromotionDecision("rejected", ("research-only",))
            if second_research_override
            else PromotionDecision("promoted", ())
        ),
        research_override=second_research_override,
    )
    items = (first, second)
    batch = resolve_dpct_batch_v1(
        reference=reference,
        sources=tuple(item[0] for item in items),
        outcomes=tuple(item[1] for item in items),
        adjudications=tuple((item[2], item[3]) for item in items),
    )
    policy = CoreNumericGuardPolicyV1(
        max_out_of_gamut_fraction=1.0,
        max_clipping_fraction=1.0,
        max_projected_fraction=1.0,
        max_new_boundary_fraction=1.0,
    )
    decisions = tuple(
        guard_core_candidate_numeric_v1(
            source=item[0],
            reference=reference,
            candidate=item[1],
            acceptance=item[2],
            admission=item[3],
            policy=policy,
        )
        for item in items
    )
    numeric = guard_core_numeric_batch_v1(
        batch=batch,
        decisions=decisions,
    )
    return reference, items, batch, decisions, numeric


def test_promoted_nonresearch_batch_is_only_authorized_for_staging() -> None:
    _reference, items, batch, _decisions, numeric = _pipeline()
    result = authorize_core_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        acceptances=tuple(item[2] for item in items),
    )
    assert result.schema_id == CORE_PRODUCT_AUTHORIZATION_SCHEMA_ID
    assert result.state == "authorized-for-staging"
    assert all(
        row.action == "authorized-for-staging" and not row.reasons
        for row in result.sources
    )
    encoded = core_product_authorization_to_json(result)
    assert "applied" not in encoded
    assert "committed" in encoded
    assert "pixels" not in encoded


def test_research_override_forces_whole_batch_fallback() -> None:
    _reference, items, batch, _decisions, numeric = _pipeline(
        second_research_override=True
    )
    assert batch.atomic_state == "pending-product-guard"
    assert numeric.atomic_state == "eligible-for-transaction"
    result = authorize_core_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        acceptances=tuple(item[2] for item in items),
    )
    assert result.state == "identity-fallback"
    assert result.sources[0].action == "authorized-for-staging"
    assert result.sources[1].reasons == (
        "algorithm-not-promoted",
        "research-baseline-override",
    )


def test_numeric_fallback_short_circuits_authorization() -> None:
    reference, items, batch, _decisions, _numeric = _pipeline()
    strict_decisions = tuple(
        guard_core_candidate_numeric_v1(
            source=item[0],
            reference=reference,
            candidate=item[1],
            acceptance=item[2],
            admission=item[3],
        )
        for item in items
    )
    numeric = guard_core_numeric_batch_v1(
        batch=batch,
        decisions=strict_decisions,
    )
    assert numeric.atomic_state == "identity-fallback"
    result = authorize_core_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
    )
    assert result.state == "identity-fallback"
    assert all(row.action == "not-evaluated" for row in result.sources)
    with pytest.raises(
        ReferenceMatchContractError,
        match="cannot authorize acceptances",
    ):
        authorize_core_product_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            acceptances=tuple(item[2] for item in items),
        )


def test_acceptance_order_and_batch_binding_are_exact() -> None:
    _reference, items, batch, _decisions, numeric = _pipeline()
    acceptances = tuple(item[2] for item in items)
    with pytest.raises(
        ReferenceMatchContractError,
        match="does not bind eligible batch row",
    ):
        authorize_core_product_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            acceptances=tuple(reversed(acceptances)),
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="one acceptance per source",
    ):
        authorize_core_product_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            acceptances=acceptances[:1],
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="does not bind the D-PCT batch",
    ):
        _r, _i, other_batch, _d, _n = _pipeline(
            second_research_override=True
        )
        authorize_core_product_staging_v1(
            batch=other_batch,
            numeric_guard=numeric,
            acceptances=acceptances,
        )


def test_roundtrip_schema_and_repeat_are_exact() -> None:
    _reference, items, batch, _decisions, numeric = _pipeline()
    kwargs = {
        "batch": batch,
        "numeric_guard": numeric,
        "acceptances": tuple(item[2] for item in items),
    }
    one = authorize_core_product_staging_v1(**kwargs)
    two = authorize_core_product_staging_v1(**kwargs)
    assert one == two
    encoded = core_product_authorization_to_json(one)
    assert core_product_authorization_from_json(encoded) == one
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, state="applied"),
            "state is unsupported",
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
                        research_baseline_override=True,
                    ),
                    value.sources[1],
                ),
            ),
            "source state is inconsistent",
        ),
        (
            lambda value: replace(
                value,
                claim_ceiling="delivery-authorized",
            ),
            "claim ceiling mismatch",
        ),
        (
            lambda value: replace(value, authorization_id="0" * 64),
            "authorization_id mismatch",
        ),
    ],
)
def test_mutations_fail_closed(mutation, message) -> None:
    _reference, items, batch, _decisions, numeric = _pipeline()
    result = authorize_core_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        acceptances=tuple(item[2] for item in items),
    )
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_core_product_staging_authorization_v1(mutation(result))


def test_unknown_json_field_fails_closed() -> None:
    _reference, items, batch, _decisions, numeric = _pipeline()
    result = authorize_core_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        acceptances=tuple(item[2] for item in items),
    )
    payload = deepcopy(result.to_dict())
    payload["surprise"] = True
    with pytest.raises(ReferenceMatchContractError, match="keys mismatch"):
        core_product_authorization_from_json(json.dumps(payload))
