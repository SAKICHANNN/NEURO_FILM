from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_p226_r1cv_rec2100_pq_runtime_compatibility import (
    build_p226_parity_fixture,
)
from src.preprocess.aces2_p3d65_canonical_pq_png import (
    publish_acescg_p3d65_1000nit_canonical_pq_png_v1,
)
from src.preprocess.opencv_rec2100_pq_decode import (
    OpenCvRec2100PqDecodeError,
    decode_validated_rec2100_pq_rgb16_png_opencv_v1,
)


def _publish(path: Path) -> np.ndarray:
    fixture = build_p226_parity_fixture().reshape(29, 34, 3)
    _, samples, _ = publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
        fixture, path
    )
    return samples


def _replace_cicp(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    offset = 8
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = bytes(payload[offset + 4 : offset + 8])
        if kind == b"cICP":
            body_start = offset + 8
            payload[body_start] = 1
            crc = zlib.crc32(kind + payload[body_start : body_start + length]) & 0xFFFFFFFF
            payload[body_start + length : body_start + length + 4] = struct.pack(">I", crc)
            path.write_bytes(payload)
            return
        offset += 12 + length
    raise AssertionError("fixture lacks cICP")


def test_p239_file_and_memory_decode_exact_rgb16(tmp_path: Path) -> None:
    source = tmp_path / "p238.png"
    expected = _publish(source)
    before = source.read_bytes()
    file_rgb, file_sha = decode_validated_rec2100_pq_rgb16_png_opencv_v1(
        source, width=34, height=29, mode="file"
    )
    memory_rgb, memory_sha = decode_validated_rec2100_pq_rgb16_png_opencv_v1(
        source, width=34, height=29, mode="memory"
    )
    np.testing.assert_array_equal(file_rgb, expected)
    np.testing.assert_array_equal(memory_rgb, expected)
    assert file_sha == memory_sha
    assert source.read_bytes() == before


def test_p239_rejects_wrong_cicp_before_decode(tmp_path: Path) -> None:
    source = tmp_path / "wrong-cicp.png"
    _publish(source)
    _replace_cicp(source)
    with pytest.raises(OpenCvRec2100PqDecodeError, match="strict"):
        decode_validated_rec2100_pq_rgb16_png_opencv_v1(
            source, width=34, height=29, mode="file"
        )


def test_p239_rejects_crc_and_truncation_before_decode(tmp_path: Path) -> None:
    valid = tmp_path / "valid.png"
    _publish(valid)
    corrupted = tmp_path / "crc.png"
    data = bytearray(valid.read_bytes())
    data[-5] ^= 1
    corrupted.write_bytes(data)
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(valid.read_bytes()[:-7])
    for source in (corrupted, truncated):
        with pytest.raises(OpenCvRec2100PqDecodeError, match="strict"):
            decode_validated_rec2100_pq_rgb16_png_opencv_v1(
                source, width=34, height=29, mode="memory"
            )


def test_p239_rejects_invalid_mode_after_strict_validation(tmp_path: Path) -> None:
    source = tmp_path / "valid.png"
    _publish(source)
    with pytest.raises(OpenCvRec2100PqDecodeError, match="mode"):
        decode_validated_rec2100_pq_rgb16_png_opencv_v1(
            source, width=34, height=29, mode="other"  # type: ignore[arg-type]
        )
