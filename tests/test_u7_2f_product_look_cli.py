from __future__ import annotations

import hashlib
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
    render_product_look_rgb,
    replay_style_safe_recipe_to_file,
)
from src.preprocess import (
    load_working_image,
    save_srgb8,
    working_image_to_srgb_float,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u7_2f_product_look_cli_v1.json"
LEGACY_PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
STATS = ROOT / "configs/film_color_stats.json"
GUARDS = ROOT / "configs/color_guardrails.json"


def _source(path: Path) -> None:
    y, x = np.mgrid[:47, :61]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _run(source: Path, output: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source),
            *arguments,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_contract_and_product_profile_are_additive() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for binding in contract["parent_evidence"]:
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    legacy = load_render_profile(LEGACY_PROFILE, root=ROOT)
    product = load_render_profile(PRODUCT_PROFILE, root=ROOT)
    assert product["profile_id"] == "safe-rich-product-v1"
    for style, parameters in legacy["style_parameters"].items():
        assert product["style_parameters"][style] == parameters
    assert product["style_parameters"]["generic_bw"] == legacy["style_parameters"][
        "hp5"
    ]
    assert hashlib.sha256(LEGACY_PROFILE.read_bytes()).hexdigest() == (
        "72a9948e6fc76e230c4593b847728712bc034840cd339e6b34f079ac7d424c79"
    )


@pytest.mark.parametrize("amount", [0.0, 0.5, 1.0])
def test_generic_bw_cli_matches_dispatcher_and_recipe_replays_exact(
    tmp_path: Path, amount: float
) -> None:
    source = tmp_path / "source.png"
    label = str(amount).replace(".", "_")
    output = tmp_path / f"generic_{label}.png"
    direct = tmp_path / f"direct_{label}.png"
    replay = tmp_path / f"replay_{label}.png"
    _source(source)
    completed = _run(
        source,
        output,
        "--style",
        "generic_bw",
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--look-amount",
        str(amount),
        "--tile-size",
        "23",
        "--tile-workers",
        "2",
        "--write-recipe",
    )
    assert completed.returncode == 0, completed.stderr

    profile = load_render_profile(PRODUCT_PROFILE, root=ROOT)
    statistics = json.loads(STATS.read_text(encoding="utf-8"))["styles"]
    encoded = working_image_to_srgb_float(load_working_image(source))
    expected = render_product_look_rgb(
        encoded,
        profile=profile,
        look_id="generic_bw",
        look_amount=amount,
        style_statistics=statistics,
        guardrails={"hp5": load_guardrail_config(GUARDS, "hp5")},
        seed=7,
        tile_size=23,
        tile_workers=2,
    )
    save_srgb8(expected, direct)
    assert output.read_bytes() == direct.read_bytes()

    recipe_path = output.with_suffix(".recipe.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["schema_id"] == RECIPE_SCHEMA_ID_V2
    assert recipe["profile"]["profile_id"] == "safe-rich-product-v1"
    assert recipe["render"]["style"] == "generic_bw"
    assert recipe["render"]["look_amount"] == amount
    assert recipe["render"]["effects"]["grain"]["color"] is False
    assert recipe["claim"]["output_label"] == "film-inspired"
    assert recipe["claim"]["evidence_grade"] == "look-approximation"

    output.unlink()
    replay_style_safe_recipe_to_file(
        recipe,
        profile_path=PRODUCT_PROFILE,
        output_path=replay,
        root=ROOT,
        tile_size=23,
    )
    assert replay.read_bytes() == direct.read_bytes()


def test_generic_bw_requires_explicit_product_profile(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "generic.png"
    _source(source)
    no_profile = _run(source, output, "--style", "generic_bw")
    assert no_profile.returncode != 0
    assert "requires safe_lab --use-render-profile" in no_profile.stderr
    assert not output.exists()

    legacy_profile = _run(
        source,
        output,
        "--style",
        "generic_bw",
        "--use-render-profile",
        "--render-profile",
        str(LEGACY_PROFILE),
    )
    assert legacy_profile.returncode != 0
    assert "does not contain style 'generic_bw'" in legacy_profile.stderr
    assert not output.exists()


def test_existing_colour_look_is_exact_between_legacy_and_product_profiles(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    legacy = tmp_path / "legacy.png"
    product = tmp_path / "product.png"
    _source(source)
    common = (
        "--style",
        "velvia_50",
        "--use-render-profile",
        "--look-amount",
        "0.5",
        "--tile-size",
        "23",
        "--tile-workers",
        "2",
    )
    first = _run(
        source,
        legacy,
        *common,
        "--render-profile",
        str(LEGACY_PROFILE),
    )
    second = _run(
        source,
        product,
        *common,
        "--render-profile",
        str(PRODUCT_PROFILE),
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert legacy.read_bytes() == product.read_bytes()


def test_existing_colour_zero_amount_recipe_is_exact_identity_and_replays(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "velvia_zero.png"
    replay = tmp_path / "velvia_zero_replay.png"
    _source(source)
    completed = _run(
        source,
        output,
        "--style",
        "velvia_50",
        "--use-render-profile",
        "--render-profile",
        str(LEGACY_PROFILE),
        "--look-amount",
        "0",
        "--write-recipe",
    )
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(
        np.asarray(Image.open(output)), np.asarray(Image.open(source))
    )
    expected_bytes = output.read_bytes()
    recipe = json.loads(output.with_suffix(".recipe.json").read_text(encoding="utf-8"))
    output.unlink()
    replay_style_safe_recipe_to_file(
        recipe,
        profile_path=LEGACY_PROFILE,
        output_path=replay,
        root=ROOT,
    )
    assert replay.read_bytes() == expected_bytes
