from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.replayable_scene_linear_npy import (
    ReplayableSceneLinearNpyError,
    ReplayableSceneLinearNpyRows,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_npy_rows_are_exact_bounded_and_file_bound(tmp_path: Path) -> None:
    pixels = np.linspace(0, 1, 17 * 23 * 3, dtype=np.float32).reshape(17, 23, 3)
    path = tmp_path / "scene.npy"
    np.save(path, pixels, allow_pickle=False)
    source = ReplayableSceneLinearNpyRows(
        path,
        expected_file_sha256=_sha(path),
        expected_pixel_sha256=hashlib.sha256(memoryview(pixels).cast("B")).hexdigest(),
    )
    np.testing.assert_array_equal(source.rows(3, 7), pixels[3:10])
    receipt = source.verify_file_identity()
    assert receipt["shape"] == [17, 23, 3]
    with pytest.raises(ReplayableSceneLinearNpyError, match="outside"):
        source.rows(16, 2)


def test_npy_rejects_wrong_identity_dtype_and_later_replacement(tmp_path: Path) -> None:
    path = tmp_path / "scene.npy"
    pixels = np.zeros((3, 5, 3), np.float32)
    np.save(path, pixels, allow_pickle=False)
    with pytest.raises(ReplayableSceneLinearNpyError, match="identity"):
        ReplayableSceneLinearNpyRows(
            path, expected_file_sha256="0" * 64, expected_pixel_sha256="1" * 64
        )
    bad = tmp_path / "bad.npy"
    np.save(bad, pixels.astype(np.float64), allow_pickle=False)
    with pytest.raises(ReplayableSceneLinearNpyError, match="little-endian float32"):
        ReplayableSceneLinearNpyRows(
            bad, expected_file_sha256=_sha(bad), expected_pixel_sha256="1" * 64
        )
    source = ReplayableSceneLinearNpyRows(
        path,
        expected_file_sha256=_sha(path),
        expected_pixel_sha256=hashlib.sha256(memoryview(pixels).cast("B")).hexdigest(),
    )
    source.close()
    np.save(path, np.ones_like(pixels), allow_pickle=False)
    with pytest.raises(ReplayableSceneLinearNpyError, match="changed"):
        source.verify_file_identity()
