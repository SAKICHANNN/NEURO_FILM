from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.filmfx.tiled_grain as tiled_grain_module
from src.filmfx import (
    GRAIN_BLUR_HALO,
    composite_layers,
    grain_residual_layer,
    staged_grain_residual_layer,
)


def _image(shape: tuple[int, int, int], seed: int = 20260717) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def _required_bytes(shape: tuple[int, int, int]) -> int:
    return 2 * int(np.prod(shape)) * np.dtype(np.float32).itemsize


@pytest.mark.parametrize(
    ("shape", "tile_size", "row_chunk", "strength", "seed", "color"),
    [
        ((17, 23, 3), 8, 5, 0.0, 0, True),
        ((37, 53, 3), 17, 11, 0.018, 7, True),
        ((47, 31, 3), 19, 7, 0.35, 41, False),
        ((61, 79, 3), 23, 13, 1.0, 2**31, True),
    ],
)
def test_staged_grain_is_byte_identical_to_legacy(
    tmp_path: Path, shape, tile_size, row_chunk, strength, seed, color
):
    base = _image(shape)
    legacy = grain_residual_layer(base, strength=strength, seed=seed, color=color)
    staged, metadata = staged_grain_residual_layer(
        base,
        scratch_root=tmp_path,
        scratch_budget_bytes=_required_bytes(shape),
        tile_size=tile_size,
        row_chunk=row_chunk,
        strength=strength,
        seed=seed,
        color=color,
    )
    np.testing.assert_array_equal(staged.residual, legacy.residual)
    legacy_composite = composite_layers(base, [legacy], output_margin=4)
    staged_composite = composite_layers(base, [staged], output_margin=4)
    np.testing.assert_array_equal(staged_composite, legacy_composite)
    np.testing.assert_array_equal(
        np.rint(staged_composite * 255.0).astype(np.uint8),
        np.rint(legacy_composite * 255.0).astype(np.uint8),
    )
    assert metadata.tiled.halo == GRAIN_BLUR_HALO
    assert metadata.tiled.max_expanded_shape[0] <= tile_size + 2 * GRAIN_BLUR_HALO
    assert metadata.tiled.max_expanded_shape[1] <= tile_size + 2 * GRAIN_BLUR_HALO
    assert metadata.row_chunk == row_chunk
    assert metadata.color is color
    assert metadata.peak_scratch_bytes == _required_bytes(shape)
    assert metadata.scratch_files_remaining == 0
    assert list(tmp_path.iterdir()) == []


def test_staged_grain_repeats_with_path_independent_metadata(tmp_path: Path):
    base = _image((41, 57, 3))
    kwargs = dict(
        scratch_root=tmp_path,
        scratch_budget_bytes=_required_bytes(base.shape),
        tile_size=17,
        row_chunk=9,
        strength=0.21,
        seed=77,
        color=True,
    )
    first, first_metadata = staged_grain_residual_layer(base, **kwargs)
    second, second_metadata = staged_grain_residual_layer(base, **kwargs)
    assert first.residual.tobytes() == second.residual.tobytes()
    assert first_metadata == second_metadata
    assert list(tmp_path.iterdir()) == []


def test_insufficient_budget_fails_without_scratch(tmp_path: Path):
    base = _image((17, 23, 3))
    with pytest.raises(ValueError, match="below required peak"):
        staged_grain_residual_layer(
            base,
            scratch_root=tmp_path,
            scratch_budget_bytes=_required_bytes(base.shape) - 1,
            tile_size=8,
            row_chunk=5,
        )
    assert list(tmp_path.iterdir()) == []


def test_filter_failure_cleans_private_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = _image((17, 23, 3))

    def fail_filter(*args, **kwargs):
        raise RuntimeError("injected filter failure")

    monkeypatch.setattr(tiled_grain_module, "gaussian_filter_safe", fail_filter)
    with pytest.raises(RuntimeError, match="injected filter failure"):
        staged_grain_residual_layer(
            base,
            scratch_root=tmp_path,
            scratch_budget_bytes=_required_bytes(base.shape),
            tile_size=8,
            row_chunk=5,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"strength": float("nan")},
        {"strength": -0.1},
        {"seed": -1},
        {"color": 1},
        {"tile_size": 0},
        {"row_chunk": 0},
        {"scratch_budget_bytes": -1},
        {"name": ""},
    ],
)
def test_invalid_parameters_fail_without_scratch(tmp_path: Path, overrides):
    base = _image((17, 23, 3))
    kwargs = {
        "scratch_root": tmp_path,
        "scratch_budget_bytes": _required_bytes(base.shape),
        "tile_size": 8,
        "row_chunk": 5,
        **overrides,
    }
    with pytest.raises(ValueError):
        staged_grain_residual_layer(base, **kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "base",
    [
        np.zeros((7, 9), dtype=np.float32),
        np.zeros((7, 9, 3), dtype=np.uint8),
        np.zeros((7, 9, 3), dtype=np.float64),
        np.full((7, 9, 3), np.nan, dtype=np.float32),
        np.zeros((7, 9, 1), dtype=np.float32),
    ],
)
def test_invalid_base_fails_without_scratch(tmp_path: Path, base):
    with pytest.raises(ValueError):
        staged_grain_residual_layer(
            base,
            scratch_root=tmp_path,
            scratch_budget_bytes=10_000,
            tile_size=4,
            row_chunk=3,
        )
    assert list(tmp_path.iterdir()) == []


def test_nonexistent_scratch_root_fails_closed(tmp_path: Path):
    base = _image((7, 9, 3))
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="existing directory"):
        staged_grain_residual_layer(
            base,
            scratch_root=missing,
            scratch_budget_bytes=_required_bytes(base.shape),
            tile_size=4,
            row_chunk=3,
        )
