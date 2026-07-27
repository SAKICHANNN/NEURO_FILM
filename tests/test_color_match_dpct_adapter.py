from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from src.color_match import (
    DPCT_COMPATIBILITY_PROFILE_ID,
    DPCT_PINNED_COMMIT,
    MATCH_PROFILE_DISPLAY_SRGB,
    PreparedMatchViewV1,
    PromotionDecision,
    ReferenceMatchContractError,
    adapt_dpct_candidate_v2,
    admit_core_apply_receipt,
    adjudicate_core_acceptance,
    make_match_view,
    validate_prepared_core_apply_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "zhuise_producer_contract_exact_bits_v2.json"
)
LOCK = ROOT / "configs" / "reference_match_dpct_compatibility_v2.json"
LOCK_SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_dpct_compatibility_lock_v2.schema.json"
)
INTENT_ID = "9" * 64


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepared(encoded_hex: str, *, provenance: str) -> PreparedMatchViewV1:
    wire = bytes.fromhex(encoded_hex)
    pixels = np.frombuffer(wire, dtype=">f4").astype(np.float32, copy=True)
    pixels = np.ascontiguousarray(pixels.reshape(2, 2, 3))
    pixels.flags.writeable = False
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=(2, 2, 3),
        render_bridge_id="neuro-film.dpct-conformance-input.v1",
        provenance_fingerprint=provenance,
    )
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _adapt(fixture: dict | None = None):
    fixture = _json(FIXTURE) if fixture is None else fixture
    source = _prepared(
        fixture["source_pixel_f32be_hex"],
        provenance="1" * 64,
    )
    reference = _prepared(
        fixture["reference_pixel_f32be_hex"],
        provenance="2" * 64,
    )
    adapted = adapt_dpct_candidate_v2(
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
        intent_id=INTENT_ID,
    )
    return adapted, source, reference


def test_v2_lock_is_strict_and_pins_corrected_fixture() -> None:
    lock = _json(LOCK)
    schema = _json(LOCK_SCHEMA)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(lock)
    assert lock["producer"]["commit"] == DPCT_PINNED_COMMIT
    assert lock["conformance"]["fixture_sha256"] == hashlib.sha256(
        FIXTURE.read_bytes()
    ).hexdigest()
    assert lock["conformance"]["fixture_sha256"] == (
        "60e7466d373a60835fecc910550bcc10723b497f362464d51d09b3f288dead48"
    )
    assert lock["profile_mapping"]["compatibility_profile_id"] == (
        DPCT_COMPATIBILITY_PROFILE_ID
    )
    assert not lock["verdict"]["consumer_may_directly_apply"]


def test_corrected_v2_fixture_crosses_exact_pixel_receipt_boundary() -> None:
    adapted, source, reference = _adapt()
    fixture = _json(FIXTURE)
    aliases = adapted.aliases
    assert aliases.source_view_id == fixture["source"]["view_id"]
    assert aliases.reference_view_id == fixture["reference"]["view_id"]
    assert aliases.bundle_id == fixture["transform"]["bundle_id"]
    assert aliases.diagnostics_id == fixture["diagnostics"]["diagnostics_id"]
    assert aliases.result_id == fixture["apply_result"]["result_id"]
    assert aliases.producer_commit == DPCT_PINNED_COMMIT
    assert adapted.transform.source_view_id == source.descriptor.view_id
    assert (
        adapted.transform.reference_view_id
        == reference.descriptor.view_id
    )
    assert adapted.transform.transform_id != aliases.bundle_id.removeprefix(
        "sha256:"
    )
    assert adapted.diagnostics.out_of_gamut_fraction == 0.25
    assert adapted.diagnostics.clipping_fraction == 0.25
    assert adapted.diagnostics.projected_fraction == 0.0
    assert adapted.diagnostics.confidence is None
    assert adapted.diagnostics.timing_ms == 1.25
    validate_prepared_core_apply_receipt(adapted.prepared_output)
    assert adapted.prepared_output.receipt.delivery_state == "candidate-only"
    assert (
        adapted.prepared_output.receipt.output_view.pixel_sha256
        == fixture["apply_result"]["output"]["pixel_sha256"].removeprefix(
            "sha256:"
        )
    )
    assert not adapted.prepared_output.pixels.flags.writeable


def test_v2_candidate_still_requires_acceptance_and_product_guard() -> None:
    adapted, source, reference = _adapt()
    acceptance = adjudicate_core_acceptance(
        source=source.descriptor,
        reference=reference.descriptor,
        transform=adapted.transform,
        capabilities=adapted.capabilities,
        diagnostics=adapted.diagnostics,
        promotion=PromotionDecision("promoted", ()),
    )
    admission = admit_core_apply_receipt(
        prepared=adapted.prepared_output,
        acceptance=acceptance,
        source=source.descriptor,
        reference=reference.descriptor,
        transform=adapted.transform,
        capabilities=adapted.capabilities,
        diagnostics=adapted.diagnostics,
    )
    assert admission.accepted_for_product_guard
    assert admission.guard_state == "pending-product-guard"
    assert admission.required_gates == ("A1", "A4", "A5")


def test_unpromoted_v2_candidate_remains_identity_fallback() -> None:
    adapted, source, reference = _adapt()
    acceptance = adjudicate_core_acceptance(
        source=source.descriptor,
        reference=reference.descriptor,
        transform=adapted.transform,
        capabilities=adapted.capabilities,
        diagnostics=adapted.diagnostics,
        promotion=PromotionDecision("rejected", ("not-promoted",)),
    )
    admission = admit_core_apply_receipt(
        prepared=adapted.prepared_output,
        acceptance=acceptance,
        source=source.descriptor,
        reference=reference.descriptor,
        transform=adapted.transform,
        capabilities=adapted.capabilities,
        diagnostics=adapted.diagnostics,
    )
    assert not admission.accepted_for_product_guard
    assert admission.guard_state == "identity-fallback"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["source"].update(width=3),
            "pixel_byte_length mismatch",
        ),
        (
            lambda value: value["transform"].update(
                bundle_id="sha256:" + "0" * 64
            ),
            "bundle_id mismatch",
        ),
        (
            lambda value: value["diagnostics"]["measurements"].update(
                clipping_fraction=0.0
            ),
            "diagnostics_id mismatch",
        ),
        (
            lambda value: value["apply_result"].update(source_width=3),
            "source geometry mismatch",
        ),
        (
            lambda value: value["apply_result"].update(
                disposition="applied"
            ),
            "not candidate-only",
        ),
    ],
)
def test_producer_identity_and_geometry_mutations_fail_closed(
    mutation,
    message,
) -> None:
    fixture = deepcopy(_json(FIXTURE))
    mutation(fixture)
    with pytest.raises(ReferenceMatchContractError, match=message):
        _adapt(fixture)


def test_old_diagnostics_v1_cannot_enter_v2_adapter() -> None:
    fixture = deepcopy(_json(FIXTURE))
    fixture["diagnostics"]["schema"] = "zhuise.diagnostics.v1"
    with pytest.raises(
        ReferenceMatchContractError,
        match="diagnostics schema mismatch",
    ):
        _adapt(fixture)
