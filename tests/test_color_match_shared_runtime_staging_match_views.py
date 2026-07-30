from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
import numpy as np
import pytest

from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.core_contracts import (
    MATCH_PROFILE_DISPLAY_SRGB,
    make_match_view,
)
from src.color_match.canonical import canonical_sha256
import src.color_match.shared_runtime_staging_match_views as bridge
from src.color_match.shared_runtime_staging_decode import (
    decode_runtime_qualified_shared_staging_bytes_v1,
)
from tests.test_color_match_shared_runtime_staging_decode import (
    _real_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_runtime_staging_match_view_bridge_v1.schema.json"
)


@pytest.mark.parametrize(
    ("suffix", "depth"),
    [
        (".png", 16),
        (".tiff", 16),
        (".png", 8),
        (".jpg", 8),
        (".tif", 8),
    ],
)
def test_decoded_staging_samples_become_exact_path_free_match_views(
    tmp_path: Path,
    suffix: str,
    depth: int,
) -> None:
    decoded = decode_runtime_qualified_shared_staging_bytes_v1(
        _real_snapshot(tmp_path, suffix=suffix, depth=depth)
    )
    first = bridge.prepare_runtime_staging_match_views_v1(decoded)
    second = bridge.prepare_runtime_staging_match_views_v1(decoded)

    assert first.record == second.record
    assert first.record.path_consumption_authorized is False
    assert first.record.persistent_views_authorized is False
    assert first.record.application_authorized is False
    assert first.record.delivery_authorized is False
    assert "report_path" not in first.record.to_dict()
    assert all(
        "output_path" not in output
        for output in first.record.to_dict()["outputs"]
    )
    for prepared, repeated, row in zip(
        first.views,
        second.views,
        first.record.outputs,
        strict=True,
    ):
        assert prepared.descriptor == repeated.descriptor
        np.testing.assert_array_equal(prepared.pixels, repeated.pixels)
        assert prepared.pixels.dtype == np.float32
        assert prepared.pixels.flags.c_contiguous
        assert not prepared.pixels.flags.writeable
        assert prepared.descriptor.profile_id == MATCH_PROFILE_DISPLAY_SRGB
        assert prepared.descriptor.render_bridge_id == (
            bridge.RUNTIME_STAGING_MATCH_VIEW_BRIDGE_ID
        )
        assert row.match_view_id == prepared.descriptor.view_id
        assert 0.0 <= float(np.min(prepared.pixels))
        assert float(np.max(prepared.pixels)) <= 1.0
    bridge.validate_runtime_staging_prepared_match_view_batch_v1(first)

    encoded = bridge.runtime_staging_match_view_bridge_record_to_json(
        first.record
    )
    assert (
        bridge.runtime_staging_match_view_bridge_record_from_json(encoded)
        == first.record
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(first.record.to_dict())


def test_float32_srgb_eotf_has_exact_frozen_branch_values() -> None:
    samples = np.array(
        [[[0, 10, 11], [128, 254, 255]]],
        dtype=np.uint8,
    )
    samples.flags.writeable = False
    actual = bridge._decode_srgb_samples_f32(samples, bit_depth=8)
    encoded = samples.astype(np.float32) / np.float32(255.0)
    expected = np.where(
        encoded <= np.float32(0.04045),
        encoded / np.float32(12.92),
        np.power(
            (encoded + np.float32(0.055)) / np.float32(1.055),
            np.float32(2.4),
        ),
    ).astype(np.float32)

    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 0, 0] == np.float32(0.0)
    assert actual[0, 1, 2] == np.float32(1.0)
    assert not actual.flags.writeable


def test_wrong_dtype_mutable_samples_and_nonfinite_math_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutable = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(
        ReferenceMatchContractError,
        match="bridge input",
    ):
        bridge._decode_srgb_samples_f32(mutable, bit_depth=8)
    wrong = np.zeros((1, 1, 3), dtype=np.uint16)
    wrong.flags.writeable = False
    with pytest.raises(
        ReferenceMatchContractError,
        match="bridge input",
    ):
        bridge._decode_srgb_samples_f32(wrong, bit_depth=8)

    readonly = np.full((1, 1, 3), 255, dtype=np.uint8)
    readonly.flags.writeable = False
    original = np.power

    def nonfinite(*args, **kwargs):
        result = original(*args, **kwargs)
        result[...] = np.nan
        return result

    monkeypatch.setattr(bridge.np, "power", nonfinite)
    with pytest.raises(
        ReferenceMatchContractError,
        match="invalid pixels",
    ):
        bridge._decode_srgb_samples_f32(readonly, bit_depth=8)


def test_view_pixel_and_persisted_authority_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    result = bridge.prepare_runtime_staging_match_views_v1(
        decode_runtime_qualified_shared_staging_bytes_v1(
            _real_snapshot(tmp_path, suffix=".png", depth=16)
        )
    )
    tampered_pixels = result.views[0].pixels.copy()
    tampered_pixels[0, 0, 0] = np.float32(0.5)
    tampered_pixels.flags.writeable = False
    tampered_view = replace(result.views[0], pixels=tampered_pixels)
    with pytest.raises(ReferenceMatchContractError):
        bridge.validate_runtime_staging_prepared_match_view_batch_v1(
            replace(result, views=(tampered_view, *result.views[1:]))
        )
    for mutation in (
        {"path_consumption_authorized": True},
        {"persistent_views_authorized": True},
        {"application_authorized": True},
        {"delivery_authorized": True},
        {"claim_ceiling": "applied"},
        {"source_count": 3},
    ):
        with pytest.raises(ReferenceMatchContractError):
            bridge.validate_runtime_staging_match_view_bridge_record_v1(
                replace(result.record, **mutation)
            )
    payload = result.record.to_dict()
    payload["application_authorized"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(
            json.loads(SCHEMA.read_text(encoding="utf-8"))
        ).validate(payload)


def test_self_consistent_descriptor_and_record_rewrite_cannot_replace_eotf(
    tmp_path: Path,
) -> None:
    result = bridge.prepare_runtime_staging_match_views_v1(
        decode_runtime_qualified_shared_staging_bytes_v1(
            _real_snapshot(tmp_path, suffix=".png", depth=16)
        )
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
        render_bridge_id=bridge.RUNTIME_STAGING_MATCH_VIEW_BRIDGE_ID,
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
        bridge.validate_runtime_staging_prepared_match_view_batch_v1(
            replace(
                result,
                record=forged_record,
                views=(forged_view, *result.views[1:]),
            )
        )
