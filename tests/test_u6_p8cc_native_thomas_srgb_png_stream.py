from __future__ import annotations

import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocess.output_encode import srgb_icc_profile
from src.preprocess.png_stream import StreamingSrgbPngWriter

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cc_native_thomas_srgb_png_stream_v1.json"


def test_p8cc_contract_freezes_streaming_png_semantics() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert contract["encoder"]["bit_depths"] == [8, 16]
    assert contract["encoder"]["icc_profile_bytes"] == 588
    assert contract["candidate"]["full_output_allowed"] is False
    assert contract["candidate"]["one_final_quantization_only"] is True


def test_p8cc_contract_keeps_claim_below_device_and_product() -> None:
    ceiling = json.loads(CONTRACT.read_text(encoding="utf-8"))["claim_ceiling"]
    assert "not device runtime" in ceiling
    assert "not" in ceiling and "product authorization" in ceiling


def _icc_payload(path: Path) -> bytes:
    raw = path.read_bytes()
    offset = 8
    while offset < len(raw):
        length = int.from_bytes(raw[offset : offset + 4], "big")
        kind = raw[offset + 4 : offset + 8]
        payload = raw[offset + 8 : offset + 8 + length]
        if kind == b"iCCP":
            name, remainder = payload.split(b"\x00", 1)
            assert name == b"K-MCFM sRGB"
            assert remainder[0] == 0
            return zlib.decompress(remainder[1:])
        offset += 12 + length
    raise AssertionError("PNG has no iCCP chunk")


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_streaming_png_roundtrips_exact_samples_and_icc(
    tmp_path: Path, bit_depth: int
) -> None:
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    maximum = np.iinfo(dtype).max
    values = (
        np.arange(7 * 11 * 3, dtype=np.uint32).reshape(7, 11, 3) * 997
    ) % (maximum + 1)
    values = np.ascontiguousarray(values.astype(dtype))
    path = tmp_path / f"exact-{bit_depth}.png"
    with StreamingSrgbPngWriter(path, width=11, height=7, bit_depth=bit_depth) as writer:
        writer.write_rows(0, values[:3])
        writer.write_rows(3, values[3:])
        digest = writer.finish()
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert np.array_equal(decoded[..., ::-1], values)
    assert _icc_payload(path) == srgb_icc_profile()
    import hashlib

    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_streaming_png_rejects_order_and_removes_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "partial.png"
    with (
        StreamingSrgbPngWriter(path, width=5, height=4, bit_depth=16) as writer,
        pytest.raises(ValueError, match="strictly ordered"),
    ):
        writer.write_rows(1, np.zeros((1, 5, 3), dtype=np.uint16))
    assert not path.exists()
    assert not path.with_suffix(".png.stream.tmp").exists()
