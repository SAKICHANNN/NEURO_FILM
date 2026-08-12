from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from scripts.benchmark_u6_p4fx_replayable_npy_png16_24mp import prepare_input


def test_prepare_input_is_exact_and_file_backed(tmp_path: Path) -> None:
    path = tmp_path / "scene.npy"
    receipt = prepare_input(path, 16, 24, 8)
    pixels = np.load(path, mmap_mode="r", allow_pickle=False)
    assert isinstance(pixels, np.memmap)
    assert hashlib.sha256(memoryview(pixels).cast("B")).hexdigest() == receipt["pixel_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["file_sha256"]
