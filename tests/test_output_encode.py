from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocess import save_srgb8


@pytest.mark.parametrize(
    ("suffix", "expected_format"),
    [(".png", "PNG"), (".jpg", "JPEG"), (".tiff", "TIFF")],
)
def test_save_srgb8_encoding_matches_extension_and_embeds_icc(
    tmp_path: Path, suffix: str, expected_format: str
) -> None:
    rgb = np.linspace(0.0, 1.0, 8 * 10 * 3, dtype=np.float32).reshape(8, 10, 3)
    path = tmp_path / f"render{suffix}"
    assert save_srgb8(rgb, path) == expected_format
    with Image.open(path) as image:
        assert image.format == expected_format
        assert image.mode == "RGB"
        assert image.size == (10, 8)
        assert len(image.info.get("icc_profile", b"")) > 0


def test_save_srgb8_rejects_unsupported_extension_and_nonfinite_values(tmp_path: Path) -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported output extension"):
        save_srgb8(rgb, tmp_path / "render.webp")
    rgb[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_srgb8(rgb, tmp_path / "render.png")
