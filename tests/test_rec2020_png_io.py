from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocess import (
    REC2020_SDR_CICP,
    SourceProfile,
    WorkingImage,
    convert_linear_rgb,
    inspect_input,
    linear_rec2020_to_rec2020,
    load_working_image,
    rec2020_to_linear_rec2020,
    save_rec2020_16_png,
)
from src.preprocess.output_encode import _inject_png_cicp, _png_chunk


def _working(pixels: np.ndarray, *, state: str = "display_linear") -> WorkingImage:
    return WorkingImage(
        pixels=pixels,
        working_space="linear_rec2020",
        transfer_state=state,
        source_transfer_state="display_referred",
        source_profile=SourceProfile("cicp", "test BT.2020 SDR"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=None,
        warnings=[],
    )


def _chunks(payload: bytes) -> list[tuple[bytes, bytes]]:
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    chunks = []
    offset = 8
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        crc = struct.unpack(">I", payload[offset + 8 + length : offset + 12 + length])[0]
        assert crc == zlib.crc32(kind + data) & 0xFFFFFFFF
        chunks.append((kind, data))
        offset += 12 + length
    return chunks


def _base_png(array: np.ndarray) -> bytes:
    succeeded, encoded = cv2.imencode(".png", array)
    assert succeeded
    return encoded.tobytes()


def test_bt2020_transfer_golden_and_signed_roundtrip() -> None:
    linear = np.asarray(
        [[[-0.25, 0.0, 0.018053968510807], [0.18, 1.0, 1.25]]],
        dtype=np.float32,
    )
    encoded = linear_rec2020_to_rec2020(linear)
    assert encoded[0, 0, 0] < 0
    assert encoded[0, 0, 1] == 0
    assert encoded[0, 1, 1] == pytest.approx(1.0, abs=2e-7)
    restored = rec2020_to_linear_rec2020(encoded)
    assert float(np.max(np.abs(restored - linear))) <= 2e-7


def test_rec2020_png_is_deterministic_tagged_rgb16_and_roundtrips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1402)
    pixels = rng.uniform(0.0, 1.0, size=(11, 13, 3)).astype(np.float32)
    pixels[0, 0] = [0.0, 1.0, 0.0]
    working = _working(pixels)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    assert save_rec2020_16_png(working, first) == "PNG"
    assert save_rec2020_16_png(working, second) == "PNG"
    assert first.read_bytes() == second.read_bytes()

    chunks = _chunks(first.read_bytes())
    kinds = [kind for kind, _ in chunks]
    assert kinds.count(b"cICP") == 1
    assert chunks[kinds.index(b"cICP")][1] == REC2020_SDR_CICP
    assert kinds.index(b"cICP") < kinds.index(b"IDAT")
    assert b"iCCP" not in kinds
    assert b"sRGB" not in kinds

    expected_samples = np.rint(
        linear_rec2020_to_rec2020(pixels) * 65535.0
    ).astype(np.uint16)
    stored_bgr = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
    assert stored_bgr is not None
    assert stored_bgr.dtype == np.uint16
    assert np.array_equal(stored_bgr[..., ::-1], expected_samples)

    inspection = inspect_input(first)
    restored = load_working_image(first)
    assert inspection.source_profile.kind == "cicp"
    assert inspection.hdr_metadata["cicp"] == REC2020_SDR_CICP.hex()
    assert restored.working_space == "linear_rec2020"
    assert restored.transfer_state == "display_linear"
    assert restored.bit_depth_in == 16
    assert float(np.max(np.abs(restored.pixels - pixels))) <= 5e-5
    srgb_witness = convert_linear_rgb(
        restored.pixels[:1, :1],
        source_space="linear_rec2020",
        destination_space="linear_srgb",
    )
    assert max(0.0, -float(srgb_witness.min()), float(srgb_witness.max()) - 1.0) >= 0.05


@pytest.mark.parametrize(
    ("working", "path_name", "message"),
    [
        (_working(np.zeros((2, 3, 3), np.float32), state="scene_linear"), "x.png", "display_linear"),
        (_working(np.zeros((2, 3, 3), np.float32)), "x.tiff", "requires a .png"),
    ],
)
def test_rec2020_png_output_rejects_unsupported_boundary_state(
    tmp_path: Path, working: WorkingImage, path_name: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        save_rec2020_16_png(working, tmp_path / path_name)


def test_rec2020_png_output_rejects_wrong_working_space(tmp_path: Path) -> None:
    working = _working(np.zeros((2, 3, 3), np.float32))
    working.working_space = "linear_srgb"
    with pytest.raises(ValueError, match="linear_rec2020"):
        save_rec2020_16_png(working, tmp_path / "x.png")


def test_unsupported_cicp_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.png"
    payload = _inject_png_cicp(
        _base_png(np.zeros((3, 4, 3), np.uint16)), bytes((9, 16, 0, 1))
    )
    path.write_bytes(payload)
    inspection = inspect_input(path)
    assert any(w.code == "unsupported_dynamic_range" for w in inspection.warnings)
    with pytest.raises(ValueError, match="refusing fallback"):
        load_working_image(path)


def test_cicp_detection_uses_png_signature_not_filename_suffix(tmp_path: Path) -> None:
    png_path = tmp_path / "tagged.png"
    renamed_path = tmp_path / "tagged.bin"
    save_rec2020_16_png(
        _working(np.full((3, 4, 3), 0.25, dtype=np.float32)), png_path
    )
    png_path.replace(renamed_path)
    restored = load_working_image(renamed_path)
    assert restored.working_space == "linear_rec2020"
    assert restored.source_profile.kind == "cicp"


def test_corrupt_duplicate_and_late_cicp_fail_closed(tmp_path: Path) -> None:
    base = _base_png(np.zeros((3, 4, 3), np.uint16))

    corrupt = bytearray(_inject_png_cicp(base))
    corrupt[corrupt.index(b"cICP") + 4] ^= 1
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(corrupt)
    with pytest.raises(ValueError, match="CRC mismatch"):
        inspect_input(corrupt_path)

    duplicate_path = tmp_path / "duplicate.png"
    duplicate_path.write_bytes(_inject_png_cicp(_inject_png_cicp(base)))
    with pytest.raises(ValueError, match="duplicate cICP"):
        inspect_input(duplicate_path)

    late_path = tmp_path / "late.png"
    late_path.write_bytes(base[:-12] + _png_chunk(b"cICP", REC2020_SDR_CICP) + base[-12:])
    with pytest.raises(ValueError, match="must precede IDAT"):
        inspect_input(late_path)


def test_supported_cicp_rejects_8bit_and_alpha_png(tmp_path: Path) -> None:
    eight_path = tmp_path / "eight.png"
    eight_path.write_bytes(
        _inject_png_cicp(_base_png(np.zeros((3, 4, 3), np.uint8)))
    )
    with pytest.raises(ValueError, match="limited to 16-bit RGB PNG"):
        load_working_image(eight_path)

    alpha_path = tmp_path / "alpha.png"
    alpha_path.write_bytes(
        _inject_png_cicp(_base_png(np.zeros((3, 4, 4), np.uint16)))
    )
    with pytest.raises(ValueError, match="alpha ingress"):
        load_working_image(alpha_path)
