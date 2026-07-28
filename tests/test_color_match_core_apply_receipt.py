from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest
from referencing import Registry, Resource

from src.color_match import (
    CORE_APPLY_RECEIPT_SCHEMA_ID,
    DIAGNOSTICS_SCHEMA_ID,
    MATCH_PROFILE_DISPLAY_REC2020,
    MATCH_PROFILE_DISPLAY_SRGB,
    DiagnosticsV1,
    ReferenceMatchContractError,
    core_apply_receipt_from_json,
    core_apply_receipt_to_json,
    make_capabilities,
    make_match_view,
    make_transform_bundle,
    prepare_core_apply_receipt,
    validate_core_apply_receipt,
    validate_core_apply_receipt_binding,
    validate_prepared_core_apply_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "configs" / "schemas"
SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
SHA_D = "4" * 64
CONTRACT_ID = "fixture.transform-bundle.v1"
ALGORITHM_ID = "fixture.algorithm.v1"


def _objects(*, status: str = "ok"):
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
        status=status,
        fallback_reason=None if status == "ok" else "fixture-fallback",
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
    return source, reference, transform, capabilities, diagnostics


def _pixels() -> np.ndarray:
    return np.linspace(
        -0.25,
        1.5,
        18,
        dtype=np.float32,
    ).reshape(2, 3, 3)


def _prepared():
    source, reference, transform, capabilities, diagnostics = _objects()
    prepared = prepare_core_apply_receipt(
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
        output_pixels=_pixels(),
    )
    return (
        prepared,
        source,
        reference,
        transform,
        capabilities,
        diagnostics,
    )


def test_prepare_receipt_copies_and_binds_exact_output() -> None:
    original = _pixels()
    prepared, source, _, transform, capabilities, diagnostics = _prepared()
    assert prepared.receipt.schema_id == CORE_APPLY_RECEIPT_SCHEMA_ID
    assert prepared.receipt.delivery_state == "candidate-only"
    assert prepared.receipt.transform_id == transform.transform_id
    assert prepared.receipt.source_view_id == source.view_id
    assert prepared.receipt.capability_id == capabilities.capability_id
    assert prepared.receipt.output_view.shape == source.shape
    assert prepared.receipt.output_view.profile_id == source.profile_id
    assert prepared.pixels.dtype == np.float32
    assert prepared.pixels.flags.c_contiguous
    assert not prepared.pixels.flags.writeable
    assert np.array_equal(prepared.pixels, original)
    original[:] = 0.0
    assert not np.array_equal(prepared.pixels, original)
    validate_prepared_core_apply_receipt(prepared)
    assert diagnostics.status == "ok"


def test_receipt_roundtrip_matches_strict_schema() -> None:
    prepared, *_ = _prepared()
    encoded = core_apply_receipt_to_json(prepared.receipt)
    assert core_apply_receipt_from_json(encoded) == prepared.receipt
    receipt_schema = json.loads(
        (
            SCHEMAS / "reference_core_apply_receipt_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    view_schema = json.loads(
        (
            SCHEMAS / "reference_core_match_view_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(receipt_schema)
    registry = Registry().with_resource(
        view_schema["$id"],
        Resource.from_contents(view_schema),
    )
    registry = registry.with_resource(
        (
            "https://neuro-film.local/schemas/"
            "reference_core_match_view_v1.schema.json"
        ),
        Resource.from_contents(view_schema),
    )
    Draft202012Validator(
        receipt_schema,
        registry=registry,
    ).validate(json.loads(encoded))


@pytest.mark.parametrize(
    ("pixels", "message"),
    [
        (
            np.zeros((1, 3, 3), dtype=np.float32),
            "shape must match",
        ),
        (
            np.full((2, 3, 3), np.nan, dtype=np.float32),
            "must be finite",
        ),
    ],
)
def test_invalid_output_pixels_fail_closed(pixels, message) -> None:
    source, reference, transform, capabilities, diagnostics = _objects()
    with pytest.raises(ReferenceMatchContractError, match=message):
        prepare_core_apply_receipt(
            source=source,
            reference=reference,
            transform=transform,
            capabilities=capabilities,
            diagnostics=diagnostics,
            output_pixels=pixels,
        )


def test_non_ok_diagnostics_cannot_receive_output_receipt() -> None:
    source, reference, transform, capabilities, diagnostics = _objects(
        status="unsupported"
    )
    with pytest.raises(ReferenceMatchContractError, match="finite ok"):
        prepare_core_apply_receipt(
            source=source,
            reference=reference,
            transform=transform,
            capabilities=capabilities,
            diagnostics=diagnostics,
            output_pixels=_pixels(),
        )


def test_mutated_prepared_pixels_are_detected() -> None:
    prepared, *_ = _prepared()
    prepared.pixels.flags.writeable = True
    prepared.pixels[0, 0, 0] += np.float32(0.25)
    prepared.pixels.flags.writeable = False
    with pytest.raises(
        ReferenceMatchContractError,
        match="pixel identity",
    ):
        validate_prepared_core_apply_receipt(prepared)


def test_swapped_execution_identity_is_detected() -> None:
    prepared, _, reference, transform, capabilities, diagnostics = _prepared()
    other_source = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=SHA_D,
        shape=(2, 3, 3),
        render_bridge_id="neuro-film.working-image-direct.v1",
        provenance_fingerprint=SHA_A,
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="source_view_id",
    ):
        validate_core_apply_receipt_binding(
            prepared.receipt,
            source=other_source,
            reference=reference,
            transform=transform,
            capabilities=capabilities,
            diagnostics=diagnostics,
        )


def test_diagnostics_swap_is_detected() -> None:
    prepared, source, reference, transform, capabilities, diagnostics = (
        _prepared()
    )
    changed = replace(diagnostics, timing_ms=3.0)
    with pytest.raises(
        ReferenceMatchContractError,
        match="execution identity mismatch",
    ):
        validate_core_apply_receipt_binding(
            prepared.receipt,
            source=source,
            reference=reference,
            transform=transform,
            capabilities=capabilities,
            diagnostics=changed,
        )


def test_profile_change_is_rejected_even_with_recomputed_identity() -> None:
    prepared, source, reference, transform, capabilities, diagnostics = (
        _prepared()
    )
    changed_view = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_REC2020,
        pixel_sha256=prepared.receipt.output_view.pixel_sha256,
        shape=source.shape,
        render_bridge_id=prepared.receipt.output_view.render_bridge_id,
        provenance_fingerprint=(
            prepared.receipt.output_view.provenance_fingerprint
        ),
    )
    changed = replace(
        prepared.receipt,
        output_view=changed_view,
        receipt_id="0" * 64,
    )
    from src.color_match.canonical import canonical_sha256

    payload = changed.to_dict()
    payload.pop("receipt_id")
    changed = replace(changed, receipt_id=canonical_sha256(payload))
    validate_core_apply_receipt(changed)
    with pytest.raises(
        ReferenceMatchContractError,
        match="profile does not match source",
    ):
        validate_core_apply_receipt_binding(
            changed,
            source=source,
            reference=reference,
            transform=transform,
            capabilities=capabilities,
            diagnostics=diagnostics,
        )


def test_parser_rejects_unknown_fields_and_applied_state() -> None:
    prepared, *_ = _prepared()
    payload = prepared.receipt.to_dict()
    payload["surprise"] = True
    with pytest.raises(ReferenceMatchContractError, match="keys mismatch"):
        core_apply_receipt_from_json(json.dumps(payload))
    with pytest.raises(
        ReferenceMatchContractError,
        match="candidate-only",
    ):
        validate_core_apply_receipt(
            replace(prepared.receipt, delivery_state="applied")
        )


def test_parser_wraps_nonfinite_nested_view() -> None:
    prepared, *_ = _prepared()
    payload = prepared.receipt.to_dict()
    payload["output_view"]["reference_white_nits"] = float("nan")
    with pytest.raises(
        ReferenceMatchContractError,
        match="not valid JSON",
    ):
        core_apply_receipt_from_json(json.dumps(payload))
