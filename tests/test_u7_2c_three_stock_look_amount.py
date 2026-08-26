from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.pipeline_color_baseline import load_guardrail_config
from src.inference import (
    RECIPE_SCHEMA_ID_V2,
    load_render_profile,
    render_resolved_safe_lab_rgb,
    replay_style_safe_recipe_to_file,
)
from src.inference.three_stock_look import (
    list_three_stock_looks,
    render_three_stock_look_rgb,
    resolve_three_stock_look_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
STATS = ROOT / "configs/film_color_stats.json"
GUARDS = ROOT / "configs/color_guardrails.json"


@pytest.fixture(scope="module")
def source() -> np.ndarray:
    y, x = np.mgrid[:67, :91]
    return np.ascontiguousarray(
        np.stack(
            (
                ((x * 13 + y * 7) % 251) / 250.0,
                ((x * 3 + y * 17 + 19) % 251) / 250.0,
                ((x * 11 + y * 5 + 43) % 251) / 250.0,
            ),
            axis=-1,
        ).astype(np.float32)
    )


def _inputs(stock_id: str):
    profile = load_render_profile(PROFILE, root=ROOT)
    row = {item["film_stock_id"]: item for item in list_three_stock_looks()}[stock_id]
    style = row["style_id"]
    statistics = json.loads(STATS.read_text(encoding="utf-8"))["styles"][style]
    guardrails = load_guardrail_config(GUARDS, style)
    return profile, style, statistics, guardrails


def test_catalog_is_three_explicit_noncalibrated_stock_targets() -> None:
    rows = list_three_stock_looks()
    assert [row["film_stock_id"] for row in rows] == [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    assert all(
        "not a calibrated stock response" in row["claim_ceiling"]
        or "data gap" in row["claim_ceiling"]
        for row in rows
    )
    rows[0]["film_stock_id"] = "forged"
    assert list_three_stock_looks()[0]["film_stock_id"] == "fujifilm_velvia_50"


@pytest.mark.parametrize(
    "stock_id", [row["film_stock_id"] for row in list_three_stock_looks()]
)
def test_zero_is_exact_identity_and_one_is_exact_existing_baseline(
    source, stock_id: str
) -> None:
    profile, style, statistics, guardrails = _inputs(stock_id)
    zero = render_three_stock_look_rgb(
        source,
        profile=profile,
        film_stock_id=stock_id,
        look_amount=0.0,
        style_statistics=statistics,
        guardrails=guardrails,
        seed=31,
    )
    expected = render_resolved_safe_lab_rgb(
        source,
        style=style,
        style_statistics=statistics,
        style_parameters=profile["style_parameters"][style],
        guardrails=guardrails,
        seed=31,
    )
    one = render_three_stock_look_rgb(
        source,
        profile=profile,
        film_stock_id=stock_id,
        look_amount=1.0,
        style_statistics=statistics,
        guardrails=guardrails,
        seed=31,
    )
    np.testing.assert_array_equal(zero, source)
    np.testing.assert_array_equal(one, expected)


@pytest.mark.parametrize(
    "stock_id", [row["film_stock_id"] for row in list_three_stock_looks()]
)
def test_intermediate_amount_is_bounded_and_full_tiled_exact(
    source, stock_id: str
) -> None:
    profile, _, statistics, guardrails = _inputs(stock_id)
    kwargs = {
        "profile": profile,
        "film_stock_id": stock_id,
        "look_amount": 0.5,
        "style_statistics": statistics,
        "guardrails": guardrails,
        "seed": 31,
    }
    full = render_three_stock_look_rgb(source, **kwargs)
    tiled = render_three_stock_look_rgb(source, **kwargs, tile_size=29, tile_workers=2)
    np.testing.assert_array_equal(tiled, full)
    assert np.isfinite(full).all()
    assert float(full.min()) > 0.0
    assert float(full.max()) < 1.0
    assert not np.array_equal(full, source)


@pytest.mark.parametrize("amount", [-0.1, 1.1, float("nan"), True])
def test_amount_fails_closed(amount) -> None:
    profile = load_render_profile(PROFILE, root=ROOT)
    with pytest.raises(ValueError, match="look_amount"):
        resolve_three_stock_look_parameters(
            profile,
            film_stock_id="kodak_ektar_100",
            look_amount=amount,
        )


def test_unknown_stock_fails_closed() -> None:
    profile = load_render_profile(PROFILE, root=ROOT)
    with pytest.raises(ValueError, match="unsupported three-stock"):
        resolve_three_stock_look_parameters(
            profile, film_stock_id="generic_film", look_amount=0.5
        )


def test_cli_amount_recipe_is_byte_exact_replay(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "portra.png"
    replay_path = tmp_path / "portra.replay.png"
    pixels = (
        (np.arange(67 * 91 * 3, dtype=np.uint32) % 251)
        .reshape(67, 91, 3)
        .astype(np.uint8)
    )
    Image.fromarray(pixels, mode="RGB").save(source_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source_path),
            "--style",
            "portra_400",
            "--use-render-profile",
            "--look-amount",
            "0.5",
            "--tile-size",
            "29",
            "--tile-workers",
            "2",
            "--write-recipe",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    recipe_path = output_path.with_suffix(".recipe.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["schema_id"] == RECIPE_SCHEMA_ID_V2
    assert recipe["render"]["look_amount"] == 0.5
    assert recipe["render"]["color_parameters"]["strength"] == 0.175
    expected = output_path.read_bytes()
    output_path.unlink()
    replay_style_safe_recipe_to_file(
        recipe,
        profile_path=PROFILE,
        output_path=replay_path,
        root=ROOT,
        tile_size=29,
    )
    assert replay_path.read_bytes() == expected


def test_cli_rejects_unversioned_amount_override(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.fromarray(np.zeros((7, 9, 3), dtype=np.uint8), mode="RGB").save(source_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source_path),
            "--look-amount",
            "0.5",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "requires safe_lab --use-render-profile" in completed.stderr
    assert not output_path.exists()
