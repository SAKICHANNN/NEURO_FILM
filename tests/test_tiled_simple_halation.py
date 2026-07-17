from __future__ import annotations

import numpy as np
import pytest

from src.filmfx import (
    composite_layers,
    composite_simple_halation_tiled,
    halation_layer,
    simple_halation_required_halo,
)


def _image(shape: tuple[int, int, int], seed: int = 77) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def _seam_mask(shape: tuple[int, int], tile_size: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for boundary in range(tile_size, shape[0], tile_size):
        mask[boundary - 1 : boundary + 1, :] = True
    for boundary in range(tile_size, shape[1], tile_size):
        mask[:, boundary - 1 : boundary + 1] = True
    return mask


def _full(base: np.ndarray, **kwargs) -> np.ndarray:
    layer = halation_layer(base, **{key: value for key, value in kwargs.items() if key != "output_margin"})
    return composite_layers(base, [layer], output_margin=kwargs.get("output_margin", 0))


def test_default_simple_halation_halo_is_31():
    assert simple_halation_required_halo() == 31


def test_halo_uses_existing_radius_clamping():
    assert simple_halation_required_halo(min_radius=2.0, max_radius=1.0) == 7
    assert simple_halation_required_halo(min_radius=0.4, max_radius=0.0) == 3
    assert simple_halation_required_halo(min_radius=1.0, max_radius=10.5) == 33


@pytest.mark.parametrize(
    ("shape", "tile_size", "kwargs"),
    [
        ((83, 107, 3), 32, {}),
        ((49, 71, 3), 19, {"strength": 0.0, "max_radius": 2.0, "scale_count": 3}),
        ((67, 53, 3), 17, {"strength": 0.31, "min_radius": 0.4, "max_radius": 5.0, "radius_gamma": 0.6}),
        ((91, 65, 3), 23, {"threshold": 0.62, "edge_threshold": 0.2, "max_radius": 10.5, "scale_count": 9, "output_margin": 4}),
    ],
)
def test_simple_halation_tiled_composite_matches_full_frame(shape, tile_size, kwargs):
    base = _image(shape)
    full = _full(base, **kwargs)
    tiled, metadata = composite_simple_halation_tiled(base, tile_size=tile_size, **kwargs)
    error = np.abs(full - tiled)
    assert float(error.max()) <= 1e-6
    seams = _seam_mask(shape[:2], tile_size)
    if seams.any():
        assert float(error[seams].max()) <= 1e-6
    expected_halo = simple_halation_required_halo(
        min_radius=kwargs.get("min_radius", 1.1),
        max_radius=kwargs.get("max_radius", 10.0),
    )
    assert metadata.halo == expected_halo
    assert metadata.max_expanded_shape[0] <= tile_size + 2 * expected_halo
    assert metadata.max_expanded_shape[1] <= tile_size + 2 * expected_halo


def test_simple_halation_tiled_composite_repeats_byte_identically():
    base = _image((73, 89, 3))
    first, first_metadata = composite_simple_halation_tiled(base, tile_size=29, output_margin=4)
    second, second_metadata = composite_simple_halation_tiled(base, tile_size=29, output_margin=4)
    assert first.tobytes() == second.tobytes()
    assert first_metadata == second_metadata


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": float("nan")},
        {"strength": 1.01},
        {"threshold": -0.1},
        {"edge_threshold": 0.0},
        {"min_radius": 0.39},
        {"max_radius": float("inf")},
        {"max_radius": 11.0},
        {"radius_gamma": 0.0},
        {"scale_count": 2},
        {"scale_count": 3.5},
        {"output_margin": 33},
    ],
)
def test_simple_halation_tiled_composite_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        composite_simple_halation_tiled(_image((17, 23, 3)), tile_size=8, **kwargs)


@pytest.mark.parametrize(
    "base",
    [
        np.zeros((7, 9), dtype=np.float32),
        np.zeros((7, 9, 3), dtype=np.uint8),
        np.full((7, 9, 3), np.nan, dtype=np.float32),
    ],
)
def test_simple_halation_tiled_composite_rejects_invalid_rgb(base):
    with pytest.raises(ValueError):
        composite_simple_halation_tiled(base, tile_size=4)
