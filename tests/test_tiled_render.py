from __future__ import annotations

import math

import numpy as np
import pytest

from src.filmfx.fast_blur import gaussian_filter_safe
from src.inference.tiled_render import (
    TiledRenderError,
    execute_tiled_local_operator,
    plan_tile_windows,
)


def _image(shape: tuple[int, int, int], seed: int = 17) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


@pytest.mark.parametrize(
    ("shape", "tile_size", "halo"),
    [((1, 1, 3), 8, 0), ((7, 11, 3), 4, 2), ((13, 9, 1), 5, 7), ((8, 8, 4), 4, 1)],
)
def test_planner_covers_every_core_pixel_once(shape, tile_size, halo):
    windows = plan_tile_windows(shape[0], shape[1], tile_size=tile_size, halo=halo)
    coverage = np.zeros(shape[:2], dtype=np.uint8)
    for window in windows:
        coverage[window.core_y0 : window.core_y1, window.core_x0 : window.core_x1] += 1
        assert window.expanded_shape[0] <= tile_size + 2 * halo
        assert window.expanded_shape[1] <= tile_size + 2 * halo
        y0, y1, x0, x1 = window.core_offsets
        assert (y1 - y0, x1 - x0) == window.core_shape
    np.testing.assert_array_equal(coverage, np.ones(shape[:2], dtype=np.uint8))


@pytest.mark.parametrize("shape,tile_size,halo", [((1, 1, 3), 8, 4), ((7, 11, 3), 4, 0), ((9, 5, 2), 3, 2)])
def test_identity_and_pointwise_callbacks_are_bit_exact(shape, tile_size, halo):
    image = _image(shape)
    identity, _ = execute_tiled_local_operator(image, lambda tile, _: tile, tile_size=tile_size, halo=halo)
    np.testing.assert_array_equal(identity, image)

    def pointwise(tile, _):
        return np.clip(tile * np.float32(0.75) + np.float32(0.125), 0.0, 1.0).astype(tile.dtype)

    tiled, _ = execute_tiled_local_operator(image, pointwise, tile_size=tile_size, halo=halo)
    np.testing.assert_array_equal(tiled, pointwise(image, None))


def test_finite_support_gaussian_matches_full_frame_without_seams():
    image = _image((37, 53, 3))
    sigma = 1.5
    truncate = 3.0
    halo = math.ceil(sigma * truncate)

    def blur(tile, _):
        return gaussian_filter_safe(
            tile,
            sigma=(sigma, sigma, 0.0),
            truncate=truncate,
            max_direct_radius=32,
        ).astype(tile.dtype, copy=False)

    full = blur(image, None)
    tiled, metadata = execute_tiled_local_operator(image, blur, tile_size=11, halo=halo)
    error = np.abs(tiled - full)
    assert float(error.max()) <= 1e-6

    seam_mask = np.zeros(image.shape[:2], dtype=bool)
    seam_mask[10::11, :] = True
    seam_mask[:, 10::11] = True
    assert float(error[seam_mask].max()) <= 1e-6
    assert metadata.max_expanded_shape[0] <= 11 + 2 * halo
    assert metadata.max_expanded_shape[1] <= 11 + 2 * halo
    assert metadata.tile_count == math.ceil(37 / 11) * math.ceil(53 / 11)


def test_repeated_execution_is_byte_identical():
    image = _image((19, 23, 3))

    def operator(tile, window):
        offset = np.float32((window.core_y0 + window.core_x0) * 1e-6)
        return (tile + offset).astype(tile.dtype)

    first, first_metadata = execute_tiled_local_operator(image, operator, tile_size=7, halo=3)
    second, second_metadata = execute_tiled_local_operator(image, operator, tile_size=7, halo=3)
    assert first.tobytes() == second.tobytes()
    assert first_metadata == second_metadata


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image": np.zeros((3, 3), dtype=np.float32), "tile_size": 2, "halo": 0},
        {"image": np.zeros((3, 3, 3), dtype=np.uint8), "tile_size": 2, "halo": 0},
        {"image": np.zeros((0, 3, 3), dtype=np.float32), "tile_size": 2, "halo": 0},
        {"image": np.full((3, 3, 3), np.nan, dtype=np.float32), "tile_size": 2, "halo": 0},
        {"image": np.zeros((3, 3, 3), dtype=np.float32), "tile_size": 0, "halo": 0},
        {"image": np.zeros((3, 3, 3), dtype=np.float32), "tile_size": 2, "halo": -1},
    ],
)
def test_invalid_inputs_fail_closed(kwargs):
    with pytest.raises(TiledRenderError):
        execute_tiled_local_operator(operator=lambda tile, _: tile, **kwargs)


@pytest.mark.parametrize(
    "operator",
    [
        lambda tile, _: tile[..., 0],
        lambda tile, _: tile[:-1],
        lambda tile, _: tile.astype(np.float64),
        lambda tile, _: np.full_like(tile, np.inf),
        lambda tile, _: [[0.0]],
    ],
)
def test_invalid_callback_outputs_fail_closed(operator):
    with pytest.raises(TiledRenderError):
        execute_tiled_local_operator(_image((5, 7, 3)), operator, tile_size=3, halo=1)


def test_callback_cannot_mutate_the_input_view():
    image = _image((5, 7, 3))

    def mutate(tile, _):
        tile[0, 0, 0] = 0.0
        return tile

    with pytest.raises(ValueError, match="read-only"):
        execute_tiled_local_operator(image, mutate, tile_size=3, halo=1)
