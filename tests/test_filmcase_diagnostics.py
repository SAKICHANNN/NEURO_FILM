from __future__ import annotations

import numpy as np

from src.filmcase.diagnostics import chroma_speckle_diagnostics


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
        chroma_speckle_diagnostics(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((5, 4, 3), dtype=np.uint8))
