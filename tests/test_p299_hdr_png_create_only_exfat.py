from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.preprocess.png_stream import (
    StreamingRec2100PqPngWriter,
    sha256_rec2100_pq_rgb16_png_samples,
)


def _samples(height: int = 3, width: int = 5) -> np.ndarray:
    return np.arange(height * width * 3, dtype=np.uint16).reshape(height, width, 3)


def test_existing_destination_is_preserved_and_stage_removed(tmp_path: Path) -> None:
    destination = tmp_path / "existing.png"
    destination.write_bytes(b"foreign")
    writer = StreamingRec2100PqPngWriter(destination, width=5, height=3)
    writer.write_rows(0, _samples())
    with pytest.raises(FileExistsError):
        writer.finish()
    assert destination.read_bytes() == b"foreign"
    assert not writer.temporary.exists()


def test_late_destination_is_preserved_and_stage_removed(tmp_path: Path) -> None:
    destination = tmp_path / "late.png"
    writer = StreamingRec2100PqPngWriter(destination, width=5, height=3)
    writer.write_rows(0, _samples())
    destination.write_bytes(b"late foreign")
    with pytest.raises(FileExistsError):
        writer.finish()
    assert destination.read_bytes() == b"late foreign"
    assert not writer.temporary.exists()


def test_successful_create_only_png_retains_exact_samples(tmp_path: Path) -> None:
    samples = _samples()
    destination = tmp_path / "created.png"
    writer = StreamingRec2100PqPngWriter(destination, width=5, height=3)
    writer.write_rows(0, samples[:2])
    writer.write_rows(2, samples[2:])
    reported = writer.finish()
    assert reported == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert (
        sha256_rec2100_pq_rgb16_png_samples(destination, width=5, height=3)
        == hashlib.sha256(samples.tobytes()).hexdigest()
    )
    assert not writer.temporary.exists()
