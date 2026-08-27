from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.preprocess.canonical_pq_png import CanonicalStreamingRec2100PqPngWriter
from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples


def _samples(height: int = 5, width: int = 7) -> np.ndarray:
    return np.arange(height * width * 3, dtype=np.uint16).reshape(height, width, 3)


def _complete(writer: CanonicalStreamingRec2100PqPngWriter) -> None:
    samples = _samples()
    writer.write_rows(0, np.ascontiguousarray(samples[:2]))
    writer.write_rows(2, np.ascontiguousarray(samples[2:]))


def test_canonical_writer_preserves_existing_destination_and_removes_stage(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "existing.png"
    destination.write_bytes(b"foreign")
    writer = CanonicalStreamingRec2100PqPngWriter(destination, width=7, height=5)
    _complete(writer)
    with pytest.raises(FileExistsError):
        writer.finish()
    assert destination.read_bytes() == b"foreign"
    assert not writer.temporary.exists()


def test_canonical_writer_preserves_late_destination_and_removes_stage(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "late.png"
    writer = CanonicalStreamingRec2100PqPngWriter(destination, width=7, height=5)
    _complete(writer)
    destination.write_bytes(b"late foreign")
    with pytest.raises(FileExistsError):
        writer.finish()
    assert destination.read_bytes() == b"late foreign"
    assert not writer.temporary.exists()


def test_canonical_writer_success_retains_exact_samples(tmp_path: Path) -> None:
    samples = _samples()
    destination = tmp_path / "created.png"
    writer = CanonicalStreamingRec2100PqPngWriter(destination, width=7, height=5)
    _complete(writer)
    reported = writer.finish()
    assert reported == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert (
        sha256_rec2100_pq_rgb16_png_samples(destination, width=7, height=5)
        == hashlib.sha256(samples.tobytes()).hexdigest()
    )
    assert not writer.temporary.exists()
