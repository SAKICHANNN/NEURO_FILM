from __future__ import annotations

import numpy as np

from src.filmcase.diagnostics import (
    chroma_speckle_diagnostics,
    structural_render_diagnostics,
)


def test_chroma_speckle_diagnostic_flags_new_high_frequency_red_islands() -> None:
    before = np.full((64, 64, 3), [120, 65, 45], dtype=np.uint8)
    after = before.copy()
    after[::4, ::4] = [255, 0, 0]
    report = chroma_speckle_diagnostics(before, after)
    assert report["diagnostic_only"] is True
    assert report["speckle_candidate_pixel_count"] > 0
    assert report["small_island_count"] > 0


def test_chroma_speckle_diagnostic_rejects_shape_mismatch() -> None:
    with np.testing.assert_raises(ValueError):
        chroma_speckle_diagnostics(
            np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((5, 4, 3), dtype=np.uint8)
        )


def test_structural_diagnostic_identity_is_exact() -> None:
    y, x = np.mgrid[:48, :64]
    image = np.stack(
        (
            (x * 3 + y * 2) % 256,
            (x * 7 + y) % 256,
            (x + y * 11) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    report = structural_render_diagnostics(image, image)
    assert report["diagnostic_only"] is True
    assert report["edge_direction_cosine_p05"] > 0.99999
    assert report["edge_gain_p05"] == 1.0
    assert report["edge_gain_median"] == 1.0
    assert report["edge_gain_p95"] == 1.0
    assert report["texture_gain_median"] == 1.0
    assert report["flat_new_high_frequency_max"] == 0.0


def test_structural_diagnostic_finds_flat_region_false_texture() -> None:
    y, x = np.mgrid[:64, :64]
    before = np.stack((x * 2 + y, x * 2 + y, x * 2 + y), axis=2).astype(np.uint8)
    after = before.copy()
    after[::2, ::2, 0] = np.minimum(after[::2, ::2, 0] + 60, 255)
    report = structural_render_diagnostics(before, after)
    assert report["flat_new_high_frequency_p99"] > 0.0
    assert report["edge_direction_cosine_p05"] < 1.0


def test_structural_diagnostic_rejects_invalid_input() -> None:
    with np.testing.assert_raises(ValueError):
        structural_render_diagnostics(
            np.zeros((8, 8, 3), dtype=np.float32),
            np.full((8, 8, 3), np.nan, dtype=np.float32),
        )
