from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.inference.three_stock_preview import (
    render_three_stock_previews_to_directory,
)
from src.preprocess import load_jpeg_preview_working_image, load_working_image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_20g_exif_oriented_jpeg_preview_v1.json"


def _oriented_jpeg(path: Path, orientation: int) -> None:
    height, width = 240, 320
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[: height // 2, : width // 2] = (235, 24, 18)
    rgb[: height // 2, width // 2 :] = (24, 220, 35)
    rgb[height // 2 :, : width // 2] = (20, 48, 230)
    rgb[height // 2 :, width // 2 :] = (225, 210, 20)
    image = Image.fromarray(rgb, mode="RGB")
    exif = image.getexif()
    exif[274] = orientation
    image.save(path, quality=96, subsampling=0, exif=exif)


def _common_size(orientation: int) -> tuple[int, int]:
    return (90, 120) if orientation in {5, 6, 7, 8} else (120, 90)


def test_contract_freezes_orientation_scope_and_claim() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["orientation_values"] == list(range(1, 9))
    assert payload["axis_swap_values"] == [5, 6, 7, 8]
    assert payload["orientation_operation"] == (
        "lossless-axis-transform-after-decoder-scale"
    )
    assert payload["invalid_orientation_policy"] == "fail-closed"
    assert payload["gates"]["orientation_step_resampling_allowed"] is False
    assert "Look Approximation" in payload["claim_ceiling"]


@pytest.mark.parametrize("orientation", range(1, 9))
def test_scaled_preview_matches_full_decode_orientation(
    tmp_path: Path, orientation: int
) -> None:
    source = tmp_path / f"orientation-{orientation}.jpg"
    _oriented_jpeg(source, orientation)
    target_width, target_height = _common_size(orientation)

    scaled = load_jpeg_preview_working_image(
        source,
        target_width=target_width,
        target_height=target_height,
    )
    full = load_working_image(source)

    assert scaled.orientation_applied is True
    assert scaled.pixels.shape[1] >= target_width
    assert scaled.pixels.shape[0] >= target_height
    assert (full.pixels.shape[1] > full.pixels.shape[0]) is (
        orientation not in {5, 6, 7, 8}
    )

    scaled_common = cv2.resize(
        scaled.pixels,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    full_common = cv2.resize(
        full.pixels,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    rmse = float(np.sqrt(np.mean((scaled_common - full_common) ** 2)))
    assert rmse <= 0.025


def test_three_look_preview_uses_oriented_source_geometry(tmp_path: Path) -> None:
    source = tmp_path / "portrait-orientation-6.jpg"
    _oriented_jpeg(source, 6)

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
        include_input_preview=True,
    )

    assert (manifest["source_width"], manifest["source_height"]) == (240, 320)
    assert manifest["preview_height"] > manifest["preview_width"]
    assert manifest["decoded_height"] > manifest["decoded_width"]
    assert manifest["jpeg_scaled_decode"] is True
    assert len(manifest["rows"]) == 3
    with Image.open(manifest["input_preview"]["output_path"]) as preview:
        assert preview.height > preview.width
    for row in manifest["rows"]:
        with Image.open(row["output_path"]) as preview:
            assert preview.height > preview.width


def test_scaled_preview_rejects_invalid_exif_orientation(tmp_path: Path) -> None:
    source = tmp_path / "invalid-orientation.jpg"
    _oriented_jpeg(source, 9)
    with pytest.raises(ValueError, match="EXIF orientation"):
        load_jpeg_preview_working_image(
            source,
            target_width=120,
            target_height=90,
        )
