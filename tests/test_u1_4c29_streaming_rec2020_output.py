from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.inference.romm_rec2020_velvia_staged_native_v2 import (
    render_supported_prophoto_velvia_rec2020_staged_native_v2,
)
from src.inference.romm_rec2020_velvia_staged_streaming_v3 import (
    render_supported_prophoto_velvia_rec2020_staged_streaming_v3,
)
from src.preprocess import REC2020_SDR_CICP
from src.preprocess.png_stream import (
    StreamingRec2020PngWriter,
    sha256_rec2020_rgb16_png_samples,
)
from tests.test_u1_4c19_staged_prophoto_render import _write_source

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/prophoto_rec2020_velvia_v1.json"


def _chunks(path: Path) -> dict[bytes, list[bytes]]:
    raw = path.read_bytes()
    result: dict[bytes, list[bytes]] = {}
    offset = 8
    while offset < len(raw):
        length = int.from_bytes(raw[offset : offset + 4], "big")
        kind = raw[offset + 4 : offset + 8]
        result.setdefault(kind, []).append(raw[offset + 8 : offset + 8 + length])
        offset += 12 + length
    return result


def test_streaming_rec2020_preserves_exact_samples_and_cicp(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    expected = tmp_path / "expected.png"
    actual = tmp_path / "actual.png"
    _write_source(source)
    render_supported_prophoto_velvia_rec2020_staged_native_v2(
        source,
        expected,
        profile_path=PROFILE,
        root=ROOT,
        scratch_dir=tmp_path / "expected_scratch",
        build_dir=tmp_path / "expected_build",
        row_chunk=13,
        thread_count=4,
    )
    first = render_supported_prophoto_velvia_rec2020_staged_streaming_v3(
        source,
        actual,
        profile_path=PROFILE,
        root=ROOT,
        scratch_dir=tmp_path / "actual_scratch",
        build_dir=tmp_path / "actual_build",
        row_chunk=13,
        thread_count=4,
    )
    expected_samples = cv2.imread(str(expected), cv2.IMREAD_UNCHANGED)
    actual_samples = cv2.imread(str(actual), cv2.IMREAD_UNCHANGED)
    assert expected_samples is not None and actual_samples is not None
    assert (actual_samples == expected_samples).all()
    chunks = _chunks(actual)
    assert chunks[b"cICP"] == [REC2020_SDR_CICP]
    assert b"iCCP" not in chunks
    assert first["output"]["sha256"] == hashlib.sha256(actual.read_bytes()).hexdigest()

    replay = tmp_path / "replay.png"
    second = render_supported_prophoto_velvia_rec2020_staged_streaming_v3(
        source,
        replay,
        profile_path=PROFILE,
        root=ROOT,
        scratch_dir=tmp_path / "replay_scratch",
        build_dir=tmp_path / "replay_build",
        row_chunk=13,
        thread_count=4,
    )
    assert replay.read_bytes() == actual.read_bytes()
    assert second == first


def test_streaming_rec2020_verifier_hashes_exact_native_samples(tmp_path: Path) -> None:
    path = tmp_path / "samples.png"
    samples = np.arange(5 * 7 * 3, dtype=np.uint16).reshape(5, 7, 3)
    writer = StreamingRec2020PngWriter(path, width=7, height=5, bit_depth=16)
    writer.write_rows(0, np.ascontiguousarray(samples[:2]))
    writer.write_rows(2, np.ascontiguousarray(samples[2:]))
    writer.finish()
    assert sha256_rec2020_rgb16_png_samples(path, width=7, height=5) == hashlib.sha256(
        samples.tobytes()
    ).hexdigest()


def test_streaming_rec2020_verifier_rejects_crc_and_trailing_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "samples.png"
    samples = np.zeros((2, 3, 3), dtype=np.uint16)
    writer = StreamingRec2020PngWriter(path, width=3, height=2, bit_depth=16)
    writer.write_rows(0, samples)
    writer.finish()
    corrupted = bytearray(path.read_bytes())
    corrupted[29] ^= 1
    bad_crc = tmp_path / "bad_crc.png"
    bad_crc.write_bytes(corrupted)
    with pytest.raises(ValueError, match="CRC"):
        sha256_rec2020_rgb16_png_samples(bad_crc, width=3, height=2)
    trailing = tmp_path / "trailing.png"
    trailing.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="trailing"):
        sha256_rec2020_rgb16_png_samples(trailing, width=3, height=2)


def test_streaming_rec2020_verifier_rejects_nonzero_filter(tmp_path: Path) -> None:
    path = tmp_path / "filtered.png"
    samples = np.zeros((1, 2, 3), dtype=np.uint16)
    writer = StreamingRec2020PngWriter(path, width=2, height=1, bit_depth=16)
    writer.write_rows(0, samples)
    writer.finish()
    chunks = _chunks(path)
    decoded = bytearray(zlib.decompress(b"".join(chunks[b"IDAT"])))
    decoded[0] = 1
    replacement = zlib.compress(bytes(decoded), level=0)
    rebuilt = bytearray(path.read_bytes()[:8])
    for kind in (b"IHDR", b"cICP"):
        payload = chunks[kind][0]
        rebuilt.extend(struct.pack(">I", len(payload)))
        rebuilt.extend(kind)
        rebuilt.extend(payload)
        rebuilt.extend(struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))
    rebuilt.extend(struct.pack(">I", len(replacement)))
    rebuilt.extend(b"IDAT")
    rebuilt.extend(replacement)
    rebuilt.extend(struct.pack(">I", zlib.crc32(b"IDAT" + replacement) & 0xFFFFFFFF))
    rebuilt.extend(struct.pack(">I", 0))
    rebuilt.extend(b"IEND")
    rebuilt.extend(struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF))
    path.write_bytes(rebuilt)
    with pytest.raises(ValueError, match="filter type zero"):
        sha256_rec2020_rgb16_png_samples(path, width=2, height=1)
