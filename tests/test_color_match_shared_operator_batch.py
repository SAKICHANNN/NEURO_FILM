from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from src.color_match import (
    MATCH_PROFILE_DISPLAY_SRGB,
    PreparedMatchViewV1,
    ReferenceMatchContractError,
    make_match_view,
)
from src.color_match.shared_operator_batch import (
    MAX_REFERENCE_MATCH_BATCH_SOURCES,
    SHARED_CLAIM_CEILING,
    make_shared_reference_operator_v1,
    prepare_shared_operator_apply_v1,
    resolve_shared_operator_batch_v1,
    shared_operator_batch_from_json,
    shared_operator_batch_to_json,
    validate_shared_operator_batch_v1,
    validate_shared_reference_operator_v1,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_shared_operator_batch_v1.schema.json"
)


def _prepared(seed: float) -> PreparedMatchViewV1:
    pixels = np.full((2, 3, 3), seed, dtype=np.float32)
    pixels[0, 0] = np.asarray(
        [seed, seed + 0.01, seed + 0.02], dtype=np.float32
    )
    pixels = np.ascontiguousarray(pixels)
    wire = pixels.astype(">f4", copy=False).tobytes(order="C")
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=pixels.shape,
        render_bridge_id="neuro-film.shared-test-input.v1",
        provenance_fingerprint=hashlib.sha256(
            f"input-{seed}".encode()
        ).hexdigest(),
    )
    pixels.flags.writeable = False
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _operator(reference: PreparedMatchViewV1, marker: str = "a"):
    return make_shared_reference_operator_v1(
        reference=reference,
        compatibility_profile_id="neuro-film.future-shared-consumer.v1",
        capability_id="zhuise.rgin.cpu-reference.v1",
        producer_commit=marker * 40,
        producer_bundle_id="sha256:" + marker * 64,
        producer_reference_view_id="sha256:" + "b" * 64,
        model_fingerprint="c" * 64,
        options_sha256="d" * 64,
    )


def _apply(operator, source, index: int, marker: str):
    output = np.clip(source.pixels + 0.01, 0.0, 1.0)
    producer_source_id = hashlib.sha256(
        f"source-{marker}".encode()
    ).hexdigest()
    result_id = hashlib.sha256(f"result-{marker}".encode()).hexdigest()
    diagnostics_id = hashlib.sha256(
        f"diagnostics-{marker}".encode()
    ).hexdigest()
    return prepare_shared_operator_apply_v1(
        operator=operator,
        source_index=index,
        source=source,
        producer_source_view_id="sha256:" + producer_source_id,
        producer_apply_result_id="sha256:" + result_id,
        diagnostics_id="sha256:" + diagnostics_id,
        output_pixels=output,
    )


def _batch():
    reference = _prepared(0.2)
    sources = (_prepared(0.3), _prepared(0.4))
    operator = _operator(reference)
    applies = (
        _apply(operator, sources[0], 0, "e"),
        _apply(operator, sources[1], 1, "h"),
    )
    batch = resolve_shared_operator_batch_v1(
        operator=operator,
        reference=reference,
        sources=sources,
        applies=applies,
    )
    return reference, sources, operator, applies, batch


def test_one_reference_operator_binds_ordered_exact_outputs() -> None:
    _, sources, operator, applies, batch = _batch()
    assert batch.operator == operator
    assert batch.source_count == 2
    assert batch.claim_ceiling == SHARED_CLAIM_CEILING
    assert [row.source_index for row in batch.sources] == [0, 1]
    assert [row.source_view_id for row in batch.sources] == [
        source.descriptor.view_id for source in sources
    ]
    assert all(not prepared.pixels.flags.writeable for prepared in applies)
    assert len({row.operator_id for row in batch.sources}) == 1
    assert "applied" not in shared_operator_batch_to_json(batch)


def test_roundtrip_schema_and_repeat_are_exact() -> None:
    reference, sources, operator, applies, batch = _batch()
    repeat = resolve_shared_operator_batch_v1(
        operator=operator,
        reference=reference,
        sources=sources,
        applies=applies,
    )
    assert repeat == batch
    encoded = shared_operator_batch_to_json(batch)
    assert shared_operator_batch_from_json(encoded) == batch
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))


def test_output_is_copied_and_input_mutation_cannot_change_receipt() -> None:
    reference = _prepared(0.2)
    source = _prepared(0.3)
    operator = _operator(reference)
    output = np.array(source.pixels + 0.01, dtype=np.float32)
    prepared = prepare_shared_operator_apply_v1(
        operator=operator,
        source_index=0,
        source=source,
        producer_source_view_id="sha256:" + "e" * 64,
        producer_apply_result_id="sha256:" + "f" * 64,
        diagnostics_id="sha256:" + "1" * 64,
        output_pixels=output,
    )
    before = prepared.pixels.copy()
    output[:] = 0.0
    assert np.array_equal(prepared.pixels, before)
    assert not prepared.pixels.flags.writeable


@pytest.mark.parametrize(
    "mutation",
    [
        "mixed-operator",
        "reordered",
        "duplicate-source",
        "duplicate-result",
        "foreign-reference",
        "missing-apply",
    ],
)
def test_mixed_partial_or_reordered_batch_fails_closed(
    mutation: str,
) -> None:
    reference, sources, operator, applies, _ = _batch()
    if mutation == "mixed-operator":
        other = _operator(reference, "2")
        applies = (
            applies[0],
            replace(
                applies[1],
                receipt=replace(
                    applies[1].receipt,
                    operator_id=other.operator_id,
                ),
            ),
        )
    elif mutation == "reordered":
        applies = (applies[1], applies[0])
    elif mutation == "duplicate-source":
        sources = (sources[0], sources[0])
    elif mutation == "duplicate-result":
        applies = (
            applies[0],
            replace(
                applies[1],
                receipt=replace(
                    applies[1].receipt,
                    producer_apply_result_id=(
                        applies[0].receipt.producer_apply_result_id
                    ),
                ),
            ),
        )
    elif mutation == "foreign-reference":
        reference = _prepared(0.6)
    elif mutation == "missing-apply":
        applies = (applies[0],)
    with pytest.raises(ReferenceMatchContractError):
        resolve_shared_operator_batch_v1(
            operator=operator,
            reference=reference,
            sources=sources,
            applies=applies,
        )


def test_operator_cannot_be_relabelled_as_per_source_fit() -> None:
    reference = _prepared(0.2)
    operator = _operator(reference)
    with pytest.raises(
        ReferenceMatchContractError,
        match="semantics",
    ):
        validate_shared_reference_operator_v1(
            replace(
                operator,
                fit_semantics="source-reference-per-source",
                batch_transform_policy="per-source-bundle",
            )
        )


@pytest.mark.parametrize("mutation", ["nonfinite", "shape", "profile"])
def test_invalid_output_or_profile_fails_before_receipt(
    mutation: str,
) -> None:
    reference = _prepared(0.2)
    source = _prepared(0.3)
    operator = _operator(reference)
    output = source.pixels.copy()
    if mutation == "nonfinite":
        output[0, 0, 0] = np.nan
    elif mutation == "shape":
        output = output[:1]
    elif mutation == "profile":
        source = replace(
            source,
            descriptor=replace(
                source.descriptor,
                profile_id="neuro-film.display-linear-rec2020-d65.v1",
            ),
        )
    with pytest.raises(ReferenceMatchContractError):
        prepare_shared_operator_apply_v1(
            operator=operator,
            source_index=0,
            source=source,
            producer_source_view_id="sha256:" + "e" * 64,
            producer_apply_result_id="sha256:" + "f" * 64,
            diagnostics_id="sha256:" + "1" * 64,
            output_pixels=output,
        )


def test_batch_identity_mutation_and_unknown_json_fail_closed() -> None:
    _, _, _, _, batch = _batch()
    with pytest.raises(
        ReferenceMatchContractError,
        match="identity mismatch",
    ):
        validate_shared_operator_batch_v1(
            replace(batch, batch_id="0" * 64)
        )
    payload = json.loads(shared_operator_batch_to_json(batch))
    payload["unknown"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        shared_operator_batch_from_json(json.dumps(payload))


def test_batch_source_limit_matches_runtime_boundary() -> None:
    reference, sources, operator, applies, _ = _batch()
    oversized_sources = (sources[0],) * (
        MAX_REFERENCE_MATCH_BATCH_SOURCES + 1
    )
    oversized_applies = (applies[0],) * (
        MAX_REFERENCE_MATCH_BATCH_SOURCES + 1
    )
    with pytest.raises(ReferenceMatchContractError, match="inventory"):
        resolve_shared_operator_batch_v1(
            operator=operator,
            reference=reference,
            sources=oversized_sources,
            applies=oversized_applies,
        )

    boundary_sources = tuple(
        _prepared(0.3 + index / 1000.0)
        for index in range(MAX_REFERENCE_MATCH_BATCH_SOURCES)
    )
    boundary_applies = tuple(
        _apply(operator, source, index, f"limit-{index}")
        for index, source in enumerate(boundary_sources)
    )
    boundary = resolve_shared_operator_batch_v1(
        operator=operator,
        reference=reference,
        sources=boundary_sources,
        applies=boundary_applies,
    )
    assert boundary.source_count == MAX_REFERENCE_MATCH_BATCH_SOURCES
    oversized_payload = boundary.to_dict()
    oversized_payload["source_count"] += 1
    oversized_payload["sources"].append(oversized_payload["sources"][0])
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert list(
        Draft202012Validator(schema).iter_errors(oversized_payload)
    )

    with pytest.raises(ReferenceMatchContractError, match="source_index"):
        prepare_shared_operator_apply_v1(
            operator=operator,
            source_index=MAX_REFERENCE_MATCH_BATCH_SOURCES,
            source=sources[0],
            producer_source_view_id="sha256:" + "e" * 64,
            producer_apply_result_id="sha256:" + "f" * 64,
            diagnostics_id="sha256:" + "1" * 64,
            output_pixels=sources[0].pixels,
        )
