from __future__ import annotations

import numpy as np
import pytest

from src.filmfx import (
    build_dust_scratch_context,
    composite_dust_scratch_tiled,
    composite_layers,
    dust_scratch_alpha_window,
    dust_scratch_layer,
)
from src.filmfx.tiled_effects import DustScratchContext


def _image(shape: tuple[int, int, int], seed: int = 20260717) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def _full(base: np.ndarray, *, strength: float, seed: int, output_margin: int = 0) -> np.ndarray:
    layer = dust_scratch_layer(base.shape, strength=strength, seed=seed)
    return composite_layers(base, [layer], output_margin=output_margin)


@pytest.mark.parametrize(
    ("shape", "strength", "seed"),
    [
        ((1, 1, 3), 0.0, 0),
        ((17, 23, 3), 0.08, 7),
        ((47, 59, 3), 0.5, 19),
        ((83, 107, 3), 1.0, 2**31),
    ],
)
def test_context_counts_and_dense_alpha_match_legacy(shape, strength, seed):
    context = build_dust_scratch_context(shape, strength=strength, seed=seed)
    expected_specks = max(1, int(shape[0] * shape[1] * 0.00018 * strength * 10.0))
    expected_scratches = max(0, int(shape[1] * 0.012 * strength))
    assert context.speck_rects.shape == (expected_specks, 4)
    assert context.scratch_rects.shape == (expected_scratches, 4)
    assert not context.speck_rects.flags.writeable
    assert not context.speck_alpha.flags.writeable
    assert context.event_bytes == (
        context.speck_rects.nbytes
        + context.speck_alpha.nbytes
        + context.scratch_rects.nbytes
        + context.scratch_alpha.nbytes
    )

    legacy = dust_scratch_layer(shape, strength=strength, seed=seed)
    reconstructed = dust_scratch_alpha_window(
        context,
        y0=0,
        y1=shape[0],
        x0=0,
        x1=shape[1],
    )
    np.testing.assert_array_equal(reconstructed, legacy.alpha)


@pytest.mark.parametrize(
    ("shape", "tile_size", "strength", "seed", "output_margin"),
    [
        ((19, 31, 3), 7, 0.0, 1, 0),
        ((67, 89, 3), 17, 0.4, 7, 0),
        ((101, 73, 3), 23, 1.0, 91, 4),
        ((129, 193, 3), 32, 0.85, 2**31, 0),
    ],
)
def test_tiled_composite_is_byte_exact(shape, tile_size, strength, seed, output_margin):
    base = _image(shape)
    full = _full(base, strength=strength, seed=seed, output_margin=output_margin)
    tiled, metadata = composite_dust_scratch_tiled(
        base,
        tile_size=tile_size,
        strength=strength,
        seed=seed,
        output_margin=output_margin,
    )
    np.testing.assert_array_equal(tiled, full)
    assert metadata.tiled.halo == 0
    assert metadata.tiled.max_expanded_shape[0] <= tile_size
    assert metadata.tiled.max_expanded_shape[1] <= tile_size
    assert metadata.context_bytes > 0


def test_context_and_composite_repeat_byte_identically():
    base = _image((97, 137, 3))
    first_context = build_dust_scratch_context(base.shape, strength=0.93, seed=77)
    second_context = build_dust_scratch_context(base.shape, strength=0.93, seed=77)
    assert first_context.speck_rects.tobytes() == second_context.speck_rects.tobytes()
    assert first_context.speck_alpha.tobytes() == second_context.speck_alpha.tobytes()
    assert first_context.scratch_rects.tobytes() == second_context.scratch_rects.tobytes()
    assert first_context.scratch_alpha.tobytes() == second_context.scratch_alpha.tobytes()

    first, first_metadata = composite_dust_scratch_tiled(
        base,
        tile_size=29,
        strength=0.93,
        seed=77,
        context=first_context,
    )
    second, second_metadata = composite_dust_scratch_tiled(
        base,
        tile_size=29,
        strength=0.93,
        seed=77,
        context=second_context,
    )
    assert first.tobytes() == second.tobytes()
    assert first_metadata == second_metadata


def test_cross_tile_long_scratch_and_overlap_are_preserved():
    base = _image((257, 389, 3))
    context = build_dust_scratch_context(base.shape, strength=1.0, seed=15)
    assert context.scratch_rects.shape[0] >= 1
    assert np.max(context.scratch_rects[:, 1] - context.scratch_rects[:, 0]) > 32
    full = _full(base, strength=1.0, seed=15)
    tiled, _ = composite_dust_scratch_tiled(
        base,
        tile_size=32,
        strength=1.0,
        seed=15,
        context=context,
    )
    np.testing.assert_array_equal(tiled, full)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": float("nan")},
        {"strength": -0.1},
        {"strength": 1.01},
        {"seed": -1},
        {"seed": 1.5},
        {"tile_size": 0},
        {"output_margin": 33},
    ],
)
def test_invalid_parameters_fail_closed(kwargs):
    base = _image((17, 23, 3))
    params = {"tile_size": 8, "strength": 0.08, "seed": 7, **kwargs}
    with pytest.raises(ValueError):
        composite_dust_scratch_tiled(base, **params)


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
def test_invalid_base_fails_closed(base):
    with pytest.raises(ValueError):
        composite_dust_scratch_tiled(base, tile_size=4)


def test_context_mismatch_and_mutation_fail_closed():
    base = _image((17, 23, 3))
    context = build_dust_scratch_context(base.shape, strength=0.4, seed=7)
    with pytest.raises(ValueError, match="does not match"):
        composite_dust_scratch_tiled(
            base,
            tile_size=8,
            strength=0.5,
            seed=7,
            context=context,
        )

    mutable = DustScratchContext(
        source_shape=context.source_shape,
        strength=context.strength,
        seed=context.seed,
        speck_rects=context.speck_rects.copy(),
        speck_alpha=context.speck_alpha.copy(),
        scratch_rects=context.scratch_rects,
        scratch_alpha=context.scratch_alpha,
    )
    with pytest.raises(ValueError, match="read-only"):
        composite_dust_scratch_tiled(
            base,
            tile_size=8,
            strength=0.4,
            seed=7,
            context=mutable,
        )
