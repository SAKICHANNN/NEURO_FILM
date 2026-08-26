from __future__ import annotations

import json
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.three_stock_batch import (
    ThreeStockBatchError,
    render_three_stock_batch_to_directory,
)
from src.inference.three_stock_look import list_three_stock_looks
from src.preprocess import load_working_image

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path) -> None:
    y, x = np.mgrid[:67, :91]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _normalized_recipe(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["output"]["path"] = "OUTPUT"
    return payload


def test_one_decode_batch_matches_three_existing_cli_exports(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    candidate = tmp_path / "candidate"
    manifest = render_three_stock_batch_to_directory(
        source,
        candidate,
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        seed=31,
        tile_size=29,
        tile_workers=2,
        png_compression=0,
    )
    assert [row["film_stock_id"] for row in manifest["rows"]] == [
        row["film_stock_id"] for row in list_three_stock_looks()
    ]
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    for row in list_three_stock_looks():
        style = row["style_id"]
        output = baseline / f"{style}.png"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/render_film.py"),
                str(source),
                "--style",
                style,
                "--use-render-profile",
                "--seed",
                "31",
                "--tile-size",
                "29",
                "--tile-workers",
                "2",
                "--output-bit-depth",
                "16",
                "--png-compression",
                "0",
                "--write-recipe",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert (candidate / output.name).read_bytes() == output.read_bytes()
        assert _normalized_recipe(candidate / f"{style}.recipe.json") == _normalized_recipe(
            baseline / f"{style}.recipe.json"
        )
    assert not list(tmp_path.glob(".candidate.*.stage"))


def test_existing_output_directory_fails_before_decode(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.bin"
    marker.write_bytes(b"keep")
    with pytest.raises(ThreeStockBatchError, match="must not already exist"):
        render_three_stock_batch_to_directory(
            source,
            output,
            root=ROOT,
            profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
            statistics_path=ROOT / "configs/film_color_stats.json",
            guardrails_path=ROOT / "configs/color_guardrails.json",
        )
    assert marker.read_bytes() == b"keep"


def test_decoded_working_image_is_collectable_before_render_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    decoded_ref: weakref.ReferenceType | None = None

    def tracked_load(path: Path):
        nonlocal decoded_ref
        working = load_working_image(path)
        decoded_ref = weakref.ref(working)
        return working

    def assert_released(*args, **kwargs):
        assert decoded_ref is not None
        assert decoded_ref() is None
        return iter(())

    monkeypatch.setattr(
        "src.inference.three_stock_batch.load_working_image", tracked_load
    )
    monkeypatch.setattr(
        "src.inference.three_stock_batch.iter_three_stock_look_rgb_shared_context",
        assert_released,
    )
    manifest = render_three_stock_batch_to_directory(
        source,
        tmp_path / "candidate",
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
    )
    assert manifest["rows"] == []
