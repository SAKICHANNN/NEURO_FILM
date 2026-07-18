from __future__ import annotations

import zlib
from pathlib import Path

import pytest
from PIL import Image

from src.preprocess import inspect_input, load_working_image


EDGE_SCAN_BYTES = 4 * 1024 * 1024
GAIN_MAP_MARKER = b"http://ns.adobe.com/hdr-gain-map/1.0/"


def _jpeg_segment(code: int, payload: bytes) -> bytes:
    assert len(payload) <= 65533
    return b"\xff" + bytes([code]) + (len(payload) + 2).to_bytes(2, "big") + payload


def _middle_metadata_jpeg(path: Path, marker: bytes | None) -> None:
    Image.new("RGB", (8, 6), (32, 96, 160)).save(path)
    original = path.read_bytes()
    padding = _jpeg_segment(0xEF, b"p" * 65520) * 65
    metadata = _jpeg_segment(0xE1, marker) if marker is not None else b""
    path.write_bytes(original[:2] + padding + metadata + padding + original[2:])


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + chunk_type + payload + crc.to_bytes(4, "big")


def _inject_before_iend(path: Path, chunks: bytes) -> None:
    payload = path.read_bytes()
    iend = payload.rfind(b"\x00\x00\x00\x00IEND")
    assert iend >= 8
    path.write_bytes(payload[:iend] + chunks + payload[iend:])


def _middle_metadata_png(path: Path, marker: bytes | None) -> None:
    Image.new("RGB", (8, 6), (32, 96, 160)).save(path)
    padding = _png_chunk(b"vpAg", b"p" * (1024 * 1024)) * 5
    metadata = _png_chunk(b"tEXt", b"XML:com.adobe.xmp\x00" + marker) if marker else b""
    _inject_before_iend(path, padding + metadata + padding)


@pytest.mark.parametrize("suffix", [".jpg", ".png"])
def test_structured_middle_marker_rejects_outside_edge_sample(
    tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"middle{suffix}"
    if suffix == ".jpg":
        _middle_metadata_jpeg(path, GAIN_MAP_MARKER)
    else:
        _middle_metadata_png(path, GAIN_MAP_MARKER)
    payload = path.read_bytes()
    position = payload.index(GAIN_MAP_MARKER)
    assert position > EDGE_SCAN_BYTES
    assert len(payload) - position - len(GAIN_MAP_MARKER) > EDGE_SCAN_BYTES

    inspection = inspect_input(path)
    warning = next(w for w in inspection.warnings if w.code == "unsupported_dynamic_range")
    assert GAIN_MAP_MARKER.decode("ascii") in warning.message
    with pytest.raises(ValueError, match="refusing SDR fallback"):
        load_working_image(path)


@pytest.mark.parametrize("suffix", [".jpg", ".png"])
def test_ordinary_large_structured_fixture_still_decodes(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"ordinary{suffix}"
    if suffix == ".jpg":
        _middle_metadata_jpeg(path, None)
    else:
        _middle_metadata_png(path, None)
    inspection = inspect_input(path)
    assert not any(w.code == "unsupported_dynamic_range" for w in inspection.warnings)
    working = load_working_image(path)
    assert working.pixels.shape == (6, 8, 3)


@pytest.mark.parametrize("chunk_type", [b"zTXt", b"iTXt"])
def test_compressed_png_text_marker_rejects(tmp_path: Path, chunk_type: bytes) -> None:
    path = tmp_path / f"compressed_{chunk_type.decode('ascii')}.png"
    Image.new("RGB", (8, 6), (32, 96, 160)).save(path)
    compressed = zlib.compress(GAIN_MAP_MARKER)
    if chunk_type == b"zTXt":
        payload = b"hdr_gain_map\x00\x00" + compressed
    else:
        payload = b"XML:com.adobe.xmp\x00\x01\x00\x00\x00" + compressed
    _inject_before_iend(path, _png_chunk(chunk_type, payload))
    with pytest.raises(ValueError, match="refusing SDR fallback"):
        load_working_image(path)


def test_overexpanding_png_text_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "overexpanding.png"
    Image.new("RGB", (8, 6), (32, 96, 160)).save(path)
    payload = b"comment\x00\x00" + zlib.compress(b"x" * (1024 * 1024 + 1))
    _inject_before_iend(path, _png_chunk(b"zTXt", payload))
    inspection = inspect_input(path)
    warning = next(w for w in inspection.warnings if w.code == "unsupported_dynamic_range")
    assert "structured:png_metadata_untrusted" in warning.message
    with pytest.raises(ValueError, match="refusing SDR fallback"):
        load_working_image(path)
