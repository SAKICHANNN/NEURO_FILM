from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from src.preprocess.canonical_pq_png import CanonicalStreamingRec2100PqPngWriter
from src.preprocess.color_management import REC2100_PQ_CICP
from src.preprocess.png_stream import (
    StreamingRec2100PqPngWriter,
    sha256_rec2020_rgb16_png_samples,
    sha256_rec2100_pq_rgb16_png_samples,
)


def _chunks(path: Path) -> list[tuple[bytes, bytes]]:
    payload = path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    chunks: list[tuple[bytes, bytes]] = []
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


def test_rec2100_pq_stream_preserves_samples_and_exact_cicp(tmp_path: Path) -> None:
    samples = np.random.default_rng(1407).integers(
        0, 65536, size=(9, 13, 3), dtype=np.uint16
    )
    samples[0, 0] = [0, 32768, 65535]
    path = tmp_path / "pq.png"
    writer = StreamingRec2100PqPngWriter(path, width=13, height=9)
    writer.write_rows(0, np.ascontiguousarray(samples[:4]))
    writer.write_rows(4, np.ascontiguousarray(samples[4:]))
    writer.finish()
    chunks = _chunks(path)
    assert [kind for kind, _ in chunks][0:2] == [b"IHDR", b"cICP"]
    assert chunks[1][1] == REC2100_PQ_CICP == bytes((9, 16, 0, 1))
    assert sha256_rec2100_pq_rgb16_png_samples(
        path, width=13, height=9
    ) == hashlib.sha256(samples.tobytes()).hexdigest()
    with pytest.raises(ValueError, match="cICP"):
        sha256_rec2020_rgb16_png_samples(path, width=13, height=9)


def test_rec2100_pq_writer_fails_atomically_on_incomplete_rows(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.png"
    with StreamingRec2100PqPngWriter(path, width=3, height=2) as writer:
        writer.write_rows(0, np.zeros((1, 3, 3), dtype=np.uint16))
        with pytest.raises(ValueError, match="before all rows"):
            writer.finish()
    assert not path.exists()


def test_rec2100_pq_writer_rejects_wrong_depth_and_row_order(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="16-bit"):
        StreamingRec2100PqPngWriter(tmp_path / "8bit.png", width=2, height=1, bit_depth=8)
    writer = StreamingRec2100PqPngWriter(tmp_path / "order.png", width=2, height=1)
    with pytest.raises(ValueError, match="strictly ordered"):
        writer.write_rows(1, np.zeros((1, 2, 3), dtype=np.uint16))
    writer.abort()
    assert not (tmp_path / "order.png").exists()


def test_canonical_pq_writer_is_partition_invariant(tmp_path: Path) -> None:
    samples = np.random.default_rng(9100).integers(
        0, 65536, size=(131, 257, 3), dtype=np.uint16
    )
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    partitions = [[64, 64, 3], [3, 64, 64]]
    for path, sizes in zip(paths, partitions, strict=True):
        writer = CanonicalStreamingRec2100PqPngWriter(
            path, width=257, height=131
        )
        start = 0
        for size in sizes:
            writer.write_rows(start, np.ascontiguousarray(samples[start : start + size]))
            start += size
        writer.finish()
    assert paths[0].read_bytes() == paths[1].read_bytes()
