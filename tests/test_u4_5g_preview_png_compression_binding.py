from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.three_stock_preview import (
    ThreeStockPreviewError,
    render_three_stock_previews_to_directory,
)
from src.preprocess import save_srgb8

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path) -> None:
    y, x = np.mgrid[:83, :121]
    rgb = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _render(source: Path, output: Path, *, compression: int | None) -> dict:
    kwargs = {}
    if compression is not None:
        kwargs["png_compression"] = compression
    return render_three_stock_previews_to_directory(
        source,
        output,
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        max_preview_pixels=4_000,
        tile_size=23,
        tile_workers=1,
        **kwargs,
    )


def _decoded(path: Path) -> tuple[np.ndarray, bytes]:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        profile = image.info.get("icc_profile", b"")
    return pixels, profile


def test_preview_compression_is_effective_pixel_exact_and_manifest_bound(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    default = _render(source, tmp_path / "default", compression=None)
    level_6 = _render(source, tmp_path / "level-6", compression=6)
    level_0 = _render(source, tmp_path / "level-0", compression=0)
    level_9 = _render(source, tmp_path / "level-9", compression=9)

    assert {key: value for key, value in default.items() if key != "rows"} == {
        key: value for key, value in level_6.items() if key != "rows"
    }
    assert default["png_compression"] == level_6["png_compression"] == 6
    for default_row, six_row, zero_row, nine_row in zip(
        default["rows"],
        level_6["rows"],
        level_0["rows"],
        level_9["rows"],
        strict=True,
    ):
        assert default_row["output_sha256"] == six_row["output_sha256"]
        paths = [
            tmp_path / name / f"{default_row['style_id']}.preview.png"
            for name in ("default", "level-6", "level-0", "level-9")
        ]
        decoded = [_decoded(path) for path in paths]
        assert all(np.array_equal(decoded[0][0], value[0]) for value in decoded[1:])
        assert all(decoded[0][1] == value[1] for value in decoded[1:])
        assert zero_row["output_sha256"] != nine_row["output_sha256"]
        assert paths[3].stat().st_size <= paths[2].stat().st_size

    for name, expected in (("level-0", 0), ("level-6", 6), ("level-9", 9)):
        persisted = json.loads((tmp_path / name / "preview.json").read_text("utf-8"))
        assert persisted["png_compression"] == expected


@pytest.mark.parametrize("invalid", [-1, 10, True, 3.0])
def test_preview_compression_rejects_before_input_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: object
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    monkeypatch.setattr(
        "src.inference.three_stock_preview.inspect_input",
        lambda path: pytest.fail("input inspection must not run"),
    )
    with pytest.raises(ThreeStockPreviewError, match="png_compression"):
        _render(source, tmp_path / "output", compression=invalid)  # type: ignore[arg-type]


def test_rgb8_png_compression_rejects_non_png_destination(tmp_path: Path) -> None:
    pixels = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="requires a .png"):
        save_srgb8(pixels, tmp_path / "output.jpg", png_compression=6)
