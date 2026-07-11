from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocess import load_working_image, working_image_to_legacy_srgb8


def test_working_image_legacy_adapter_round_trips_srgb_fixture(tmp_path: Path) -> None:
    source = np.array(
        [[[0, 32, 128], [64, 192, 255]], [[12, 80, 160], [24, 120, 240]]],
        dtype=np.uint8,
    )
    path = tmp_path / "fixture.png"
    Image.fromarray(source, mode="RGB").save(path)

    working = load_working_image(path)
    adapted = np.asarray(working_image_to_legacy_srgb8(working))

    assert working.working_space == "linear_srgb"
    assert working.transfer_state == "display_linear"
    assert np.max(np.abs(adapted.astype(np.int16) - source.astype(np.int16))) <= 1


def test_working_image_legacy_adapter_rejects_wrong_color_state(tmp_path: Path) -> None:
    path = tmp_path / "fixture.png"
    Image.new("RGB", (2, 2), (128, 64, 32)).save(path)
    working = load_working_image(path)
    working.transfer_state = "unknown"

    try:
        working_image_to_legacy_srgb8(working)
    except ValueError as exc:
        assert "requires linear_srgb/display_linear" in str(exc)
    else:
        raise AssertionError("wrong color state must fail closed")
