from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.three_stock_preview import (
    ThreeStockPreviewError,
    preview_dimensions,
    render_three_stock_previews_to_directory,
)

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path, width: int = 121, height: int = 83) -> None:
    y, x = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _render(source: Path, output: Path, *, max_pixels: int = 4_000) -> dict:
    return render_three_stock_previews_to_directory(
        source,
        output,
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        max_preview_pixels=max_pixels,
        tile_size=23,
        tile_workers=2,
    )


def test_preview_dimensions_are_bounded_and_never_upsample() -> None:
    assert preview_dimensions(4_032, 6_048, 1_000_000) == (816, 1_224)
    assert preview_dimensions(40, 30, 2_000) == (40, 30)
    with pytest.raises(ThreeStockPreviewError, match="positive integer"):
        preview_dimensions(40, 30, 0)


def test_direct_preview_is_repeat_exact_ordered_and_bounded(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    first = _render(source, tmp_path / "first")
    second = _render(source, tmp_path / "second")

    assert first["preview_pixels"] <= 4_000
    source_ratio = 121 / 83
    preview_ratio = first["preview_width"] / first["preview_height"]
    assert abs(preview_ratio - source_ratio) <= 1.0 / first["preview_height"]
    assert [row["style_id"] for row in first["rows"]] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
    ]
    first_hashes = [row["output_sha256"] for row in first["rows"]]
    second_hashes = [row["output_sha256"] for row in second["rows"]]
    assert first_hashes == second_hashes
    assert len(set(first_hashes)) == 3
    persisted = json.loads((tmp_path / "first" / "preview.json").read_text("utf-8"))
    assert persisted == first


def test_existing_destination_rejects_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    destination = tmp_path / "existing"
    destination.mkdir()
    monkeypatch.setattr(
        "src.inference.three_stock_preview.load_working_image",
        lambda path: pytest.fail("decode must not run"),
    )
    with pytest.raises(ThreeStockPreviewError, match="must not already exist"):
        _render(source, destination)


def test_small_source_is_not_upsampled(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source, width=41, height=29)
    manifest = _render(source, tmp_path / "preview", max_pixels=10_000)
    assert (manifest["preview_width"], manifest["preview_height"]) == (41, 29)
