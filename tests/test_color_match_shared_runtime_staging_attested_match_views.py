from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
import numpy as np
import pytest

from src.color_match.canonical import canonical_sha256
from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.core_contracts import (
    MATCH_PROFILE_DISPLAY_SRGB,
    make_match_view,
)
import src.color_match.shared_runtime_staging_attested_match_views as bridge
from src.color_match.shared_runtime_staging_color_attestation import (
    attest_runtime_staging_srgb_metadata_v1,
)
from src.color_match.shared_runtime_staging_decode import (
    decode_runtime_qualified_shared_staging_bytes_v1,
)
from src.color_match.shared_runtime_staging_match_views import (
    prepare_runtime_staging_match_views_v1,
)
from tests.test_color_match_shared_runtime_staging_decode import (
    _real_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_runtime_staging_attested_match_view_bridge_v2.schema.json"
)


def _attested(
    tmp_path: Path,
    *,
    suffix: str,
    depth: int,
):
    decoded = decode_runtime_qualified_shared_staging_bytes_v1(
        _real_snapshot(tmp_path, suffix=suffix, depth=depth)
    )
    return attest_runtime_staging_srgb_metadata_v1(decoded)


@pytest.mark.parametrize(
    ("suffix", "depth", "binding"),
    [
        (".png", 16, "png-iccp-exact-v1"),
        (".tiff", 16, "tiff-34675-icc-exact-v1"),
        (".png", 8, "png-iccp-exact-v1"),
        (".jpg", 8, "jpeg-app2-icc-exact-v1"),
        (".tif", 8, "tiff-34675-icc-exact-v1"),
    ],
)
def test_only_attested_bytes_become_v2_match_views(
    tmp_path: Path,
    suffix: str,
    depth: int,
    binding: str,
) -> None:
    attested = _attested(
        tmp_path,
        suffix=suffix,
        depth=depth,
    )
    first = bridge.prepare_runtime_staging_attested_match_views_v2(
        attested
    )
    second = bridge.prepare_runtime_staging_attested_match_views_v2(
        attested
    )
    legacy = prepare_runtime_staging_match_views_v1(attested.decoded)

    assert first.record == second.record
    assert first.attested is attested
    assert first.record.attestation_id == attested.record.attestation_id
    assert first.record.path_consumption_authorized is False
    assert first.record.persistent_views_authorized is False
    assert first.record.application_authorized is False
    assert first.record.delivery_authorized is False
    encoded_record = json.dumps(first.record.to_dict()).lower()
    assert "report_path" not in encoded_record
    assert "output_path" not in encoded_record
    for prepared, repeated, legacy_view, row, metadata in zip(
        first.views,
        second.views,
        legacy.views,
        first.record.outputs,
        attested.record.outputs,
        strict=True,
    ):
        assert prepared.descriptor == repeated.descriptor
        np.testing.assert_array_equal(prepared.pixels, repeated.pixels)
        np.testing.assert_array_equal(prepared.pixels, legacy_view.pixels)
        assert prepared.descriptor.view_id != legacy_view.descriptor.view_id
        assert prepared.descriptor.profile_id == MATCH_PROFILE_DISPLAY_SRGB
        assert prepared.descriptor.render_bridge_id == (
            bridge.RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_ID
        )
        assert row.metadata_attestation_id == (
            metadata.metadata_attestation_id
        )
        assert row.profile_binding == binding
        assert row.profile_sha256 == metadata.profile_sha256
        assert row.match_view_id == prepared.descriptor.view_id
        assert not prepared.pixels.flags.writeable
    bridge.validate_runtime_staging_attested_prepared_match_view_batch_v2(
        first
    )


def test_v2_record_roundtrip_schema_and_authority_are_strict(
    tmp_path: Path,
) -> None:
    result = bridge.prepare_runtime_staging_attested_match_views_v2(
        _attested(tmp_path, suffix=".png", depth=16)
    )
    encoded = (
        bridge.runtime_staging_attested_match_view_bridge_record_to_json(
            result.record
        )
    )
    assert (
        bridge.runtime_staging_attested_match_view_bridge_record_from_json(
            encoded
        )
        == result.record
    )
    schema = json.loads(SCHEMA.read_text("utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(result.record.to_dict())

    for mutation in (
        {"path_consumption_authorized": True},
        {"persistent_views_authorized": True},
        {"application_authorized": True},
        {"delivery_authorized": True},
        {"claim_ceiling": "applied"},
        {"attestation_id": "0" * 64},
    ):
        with pytest.raises(ReferenceMatchContractError):
            bridge.validate_runtime_staging_attested_match_view_bridge_record_v2(
                replace(result.record, **mutation)
            )
    payload = result.record.to_dict()
    payload["application_authorized"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(payload)

    jpeg_16 = replace(
        result.record.outputs[0],
        profile_binding="jpeg-app2-icc-exact-v1",
    )
    provisional = replace(
        result.record,
        bridge_run_id="0" * 64,
        outputs=(jpeg_16, *result.record.outputs[1:]),
    )
    forged = replace(
        provisional,
        bridge_run_id=canonical_sha256(
            bridge._identity_payload(provisional)
        ),
    )
    with pytest.raises(ReferenceMatchContractError):
        bridge.validate_runtime_staging_attested_match_view_bridge_record_v2(
            forged
        )


def test_decoded_batch_cannot_bypass_attestation(tmp_path: Path) -> None:
    attested = _attested(tmp_path, suffix=".jpg", depth=8)
    with pytest.raises(ReferenceMatchContractError):
        bridge.prepare_runtime_staging_attested_match_views_v2(
            attested.decoded  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "mutation",
    ["pixels", "decoded_outputs", "snapshot_outputs", "output_bytes"],
)
def test_nested_input_graph_must_remain_immutable(
    tmp_path: Path,
    mutation: str,
) -> None:
    attested = _attested(tmp_path, suffix=".png", depth=8)
    decoded = attested.decoded
    if mutation == "pixels":
        decoded = replace(decoded, pixels=list(decoded.pixels))
    elif mutation == "decoded_outputs":
        decoded = replace(
            decoded,
            record=replace(
                decoded.record,
                outputs=list(decoded.record.outputs),
            ),
        )
    elif mutation == "snapshot_outputs":
        decoded = replace(
            decoded,
            snapshot=replace(
                decoded.snapshot,
                record=replace(
                    decoded.snapshot.record,
                    outputs=list(decoded.snapshot.record.outputs),
                ),
            ),
        )
    else:
        decoded = replace(
            decoded,
            snapshot=replace(
                decoded.snapshot,
                output_bytes=list(decoded.snapshot.output_bytes),
            ),
        )
    mutated = replace(attested, decoded=decoded)

    with pytest.raises(
        ReferenceMatchContractError,
        match="object graph must be immutable",
    ):
        bridge.prepare_runtime_staging_attested_match_views_v2(mutated)


def test_different_attestation_cannot_replace_bound_evidence(
    tmp_path: Path,
) -> None:
    first_attested = _attested(
        tmp_path / "first",
        suffix=".png",
        depth=8,
    )
    second_attested = _attested(
        tmp_path / "second",
        suffix=".png",
        depth=8,
    )
    result = bridge.prepare_runtime_staging_attested_match_views_v2(
        first_attested
    )

    with pytest.raises(ReferenceMatchContractError):
        bridge.validate_runtime_staging_attested_prepared_match_view_batch_v2(
            replace(result, attested=second_attested)
        )


def test_self_consistent_view_rewrite_cannot_replace_attested_eotf(
    tmp_path: Path,
) -> None:
    result = bridge.prepare_runtime_staging_attested_match_views_v2(
        _attested(tmp_path, suffix=".tif", depth=8)
    )
    original = result.views[0]
    pixels = original.pixels.copy()
    pixels[0, 0, 0] = np.float32(0.5)
    pixels.flags.writeable = False
    pixel_sha = bridge._float32_pixel_sha256(pixels)
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=pixel_sha,
        shape=tuple(int(value) for value in pixels.shape),
        render_bridge_id=(
            bridge.RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_ID
        ),
        provenance_fingerprint=original.descriptor.provenance_fingerprint,
        alpha_mode="absent",
    )
    forged_view = replace(
        original,
        descriptor=descriptor,
        pixels=pixels,
    )
    forged_output = replace(
        result.record.outputs[0],
        match_view_id=descriptor.view_id,
        match_view_pixel_sha256=pixel_sha,
    )
    provisional = replace(
        result.record,
        bridge_run_id="0" * 64,
        outputs=(forged_output, *result.record.outputs[1:]),
    )
    forged_record = replace(
        provisional,
        bridge_run_id=canonical_sha256(
            bridge._identity_payload(provisional)
        ),
    )

    with pytest.raises(
        ReferenceMatchContractError,
        match="does not match binding",
    ):
        bridge.validate_runtime_staging_attested_prepared_match_view_batch_v2(
            replace(
                result,
                record=forged_record,
                views=(forged_view, *result.views[1:]),
            )
        )


def test_view_permutation_and_single_metadata_substitution_fail_closed(
    tmp_path: Path,
) -> None:
    result = bridge.prepare_runtime_staging_attested_match_views_v2(
        _attested(tmp_path, suffix=".png", depth=8)
    )
    with pytest.raises(ReferenceMatchContractError):
        bridge.validate_runtime_staging_attested_prepared_match_view_batch_v2(
            replace(result, views=tuple(reversed(result.views)))
        )

    substituted = replace(
        result.record.outputs[0],
        metadata_attestation_id=(
            result.record.outputs[1].metadata_attestation_id
        ),
    )
    provisional = replace(
        result.record,
        bridge_run_id="0" * 64,
        outputs=(substituted, *result.record.outputs[1:]),
    )
    forged_record = replace(
        provisional,
        bridge_run_id=canonical_sha256(
            bridge._identity_payload(provisional)
        ),
    )
    with pytest.raises(ReferenceMatchContractError):
        bridge.validate_runtime_staging_attested_prepared_match_view_batch_v2(
            replace(result, record=forged_record)
        )


def test_json_duplicate_constants_and_mutable_outputs_fail_closed(
    tmp_path: Path,
) -> None:
    result = bridge.prepare_runtime_staging_attested_match_views_v2(
        _attested(tmp_path, suffix=".png", depth=8)
    )
    encoded = (
        bridge.runtime_staging_attested_match_view_bridge_record_to_json(
            result.record
        )
    )
    duplicate = encoded.replace(
        '"schema_id":',
        '"schema_id": "shadow", "schema_id":',
        1,
    )
    with pytest.raises(ReferenceMatchContractError, match="not valid JSON"):
        bridge.runtime_staging_attested_match_view_bridge_record_from_json(
            duplicate
        )
    with pytest.raises(ReferenceMatchContractError, match="not valid JSON"):
        bridge.runtime_staging_attested_match_view_bridge_record_from_json(
            encoded.replace("false", "NaN", 1)
        )

    mutable = replace(
        result.record,
        outputs=list(result.record.outputs),  # type: ignore[arg-type]
    )
    mutable = replace(
        mutable,
        bridge_run_id=canonical_sha256(
            bridge._identity_payload(mutable)
        ),
    )
    with pytest.raises(ReferenceMatchContractError):
        bridge.validate_runtime_staging_attested_match_view_bridge_record_v2(
            mutable
        )
