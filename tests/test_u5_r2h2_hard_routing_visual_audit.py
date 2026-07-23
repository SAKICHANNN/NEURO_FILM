from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.hard_routing_visual_audit import (
    _fit_image,
    blind_assignment,
)


def test_blind_assignment_is_stable_and_round_specific() -> None:
    first = [
        blind_assignment(20260723, "full_fit", sample_id)
        for sample_id in ("06", "11", "12", "18", "20", "23", "24", "28")
    ]
    second = [
        blind_assignment(20260723, "full_fit", sample_id)
        for sample_id in ("06", "11", "12", "18", "20", "23", "24", "28")
    ]
    other_round = [
        blind_assignment(20260723, "detail_crop", sample_id)
        for sample_id in ("06", "11", "12", "18", "20", "23", "24", "28")
    ]
    assert first == second
    assert first != other_round


def test_fit_image_has_exact_canvas_for_full_and_crop(tmp_path: Path) -> None:
    array = np.zeros((100, 160, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(160, dtype=np.uint8)[None, :]
    path = tmp_path / "input.png"
    Image.fromarray(array, mode="RGB").save(path)
    full = _fit_image(path, (80, 80), False)
    crop = _fit_image(path, (80, 80), True)
    assert full.size == (80, 80)
    assert crop.size == (80, 80)
    assert np.asarray(full).shape == (80, 80, 3)
    assert not np.array_equal(np.asarray(full), np.asarray(crop))
