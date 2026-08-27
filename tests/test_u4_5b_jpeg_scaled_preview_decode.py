from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.three_stock_preview import render_three_stock_previews_to_directory
from src.preprocess import load_jpeg_preview_working_image

ROOT = Path(__file__).resolve().parents[1]


def _jpeg(path: Path, *, width: int = 512, height: int = 384) -> None:
    y, x = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, quality=93)


def test_scaled_jpeg_preview_decodes_before_float_expansion(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    _jpeg(source)
    working = load_jpeg_preview_working_image(
        source,
        target_width=120,
        target_height=90,
    )
    assert working.pixels.shape == (96, 128, 3)
    assert working.pixels.dtype == np.float32
    assert working.working_space == "linear_srgb"
    assert any(
        warning.code == "jpeg_scaled_preview_decode" for warning in working.warnings
    )


def test_scaled_jpeg_preview_rejects_non_jpeg_and_upsampling(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    _jpeg(source)
    with pytest.raises(ValueError, match="cannot upsample"):
        load_jpeg_preview_working_image(
            source,
            target_width=513,
            target_height=384,
        )
    png = tmp_path / "source.png"
    Image.open(source).save(png)
    with pytest.raises(ValueError, match="limited to single-frame JPEG"):
        load_jpeg_preview_working_image(
            png,
            target_width=120,
            target_height=90,
        )


def test_three_stock_preview_uses_explicit_scaled_decode(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    _jpeg(source)
    manifest = render_three_stock_previews_to_directory(
        source,
        tmp_path / "preview",
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        max_preview_pixels=12_000,
        tile_size=23,
        tile_workers=1,
        jpeg_scaled_decode=True,
    )
    assert manifest["jpeg_scaled_decode"] is True
    assert manifest["decoded_width"] < manifest["source_width"]
    assert manifest["decoded_height"] < manifest["source_height"]
    assert manifest["decoded_width"] >= manifest["preview_width"]
    assert manifest["decoded_height"] >= manifest["preview_height"]
    assert manifest["preview_basis"].startswith("libjpeg scaled decode")
