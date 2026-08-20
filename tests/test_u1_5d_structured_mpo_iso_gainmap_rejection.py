from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.preprocess import inspect_input, load_working_image

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "u1_5c_libultrahdr" / "apple_gainmap_old.jpg"
APPLE_MARKER = b"urn:com:apple:photo:2020:aux:hdrgainmap"
ISO_MARKER = b"urn:iso:std:iso:ts:21496:-1"


def _jpeg_segment(code: int, payload: bytes) -> bytes:
    return b"\xff" + bytes([code]) + (len(payload) + 2).to_bytes(2, "big") + payload


def test_iso_21496_namespace_in_jpeg_rejects_before_sdr_decode(tmp_path: Path) -> None:
    path = tmp_path / "iso_gainmap.jpg"
    Image.new("RGB", (8, 6), (32, 96, 160)).save(path)
    payload = path.read_bytes()
    path.write_bytes(payload[:2] + _jpeg_segment(0xE1, ISO_MARKER) + payload[2:])

    inspection = inspect_input(path)
    warning = next(
        warning
        for warning in inspection.warnings
        if warning.code == "unsupported_dynamic_range"
    )
    assert ISO_MARKER.decode("ascii") in warning.message
    with pytest.raises(ValueError, match="HDR/gain-map reconstruction is not implemented"):
        load_working_image(path)


def test_real_apple_mpo_is_structurally_identified_before_multiframe_gate() -> None:
    inspection = inspect_input(FIXTURE)
    assert inspection.format_name == "MPO"
    assert inspection.frame_count == 2
    warning = next(
        warning
        for warning in inspection.warnings
        if warning.code == "unsupported_dynamic_range"
    )
    assert APPLE_MARKER.decode("ascii") in warning.message
    with pytest.raises(ValueError, match="HDR/gain-map reconstruction is not implemented"):
        load_working_image(FIXTURE)


def test_generic_mpo_without_recognized_gainmap_token_stays_multiframe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generic.mpo"
    payload = FIXTURE.read_bytes()
    assert APPLE_MARKER in payload
    path.write_bytes(payload.replace(APPLE_MARKER, b"x" * len(APPLE_MARKER)))

    inspection = inspect_input(path)
    assert inspection.format_name == "MPO"
    assert inspection.frame_count == 2
    assert not any(
        warning.code == "unsupported_dynamic_range" for warning in inspection.warnings
    )
    with pytest.raises(ValueError, match="multi-frame raster rendering is not implemented"):
        load_working_image(path)


def test_ordinary_sdr_jpeg_still_decodes(tmp_path: Path) -> None:
    path = tmp_path / "ordinary.jpg"
    Image.new("RGB", (8, 6), (32, 96, 160)).save(path)
    inspection = inspect_input(path)
    assert not any(
        warning.code == "unsupported_dynamic_range" for warning in inspection.warnings
    )
    working = load_working_image(path)
    assert working.pixels.shape == (6, 8, 3)
