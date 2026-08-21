from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.pipeline_color_baseline import load_guardrail_config, style_transfer_rgb
from src.inference import (
    StyleSafeEngineError,
    load_render_profile,
    render_resolved_safe_lab_rgb,
    render_style_safe_working_image,
    replay_style_safe_color_recipe,
)
from src.preprocess import load_working_image, save_srgb8, working_image_to_srgb_float

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs" / "render_profiles" / "safe_rich_v1.json"
STATS_PATH = ROOT / "configs" / "film_color_stats.json"
GUARDRAILS_PATH = ROOT / "configs" / "color_guardrails.json"


@pytest.fixture(scope="module")
def working(tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("u7_1a")
    y, x = np.mgrid[:31, :47]
    pixels = np.stack(
        (
            (x * 5 + y * 3) % 256,
            (x * 2 + y * 7 + 31) % 256,
            (x * 11 + y + 79) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    path = directory / "source.png"
    Image.fromarray(pixels, mode="RGB").save(path)
    return load_working_image(path)


@pytest.mark.parametrize(
    "style",
    [
        "ektar_100",
        "hp5",
        "portra_400",
        "portra_800",
        "tri_x_400",
        "velvia_50",
        "vision3_250d",
        "vision3_500t",
    ],
)
def test_public_engine_is_exact_legacy_float32_parity(working, style: str) -> None:
    profile = load_render_profile(PROFILE_PATH, root=ROOT)
    statistics = json.loads(STATS_PATH.read_text(encoding="utf-8"))["styles"][style]
    guardrails = load_guardrail_config(GUARDRAILS_PATH, style)
    source = working_image_to_srgb_float(working)
    frozen = working.pixels.copy()
    parameters = profile["style_parameters"][style]
    expected = style_transfer_rgb(
        source,
        statistics,
        style,
        strength=parameters["strength"],
        luma_strength=parameters["luma_strength"],
        grain=parameters["grain"],
        seed=7,
        gamut_safe=parameters["gamut_safe"],
        gamut_mode=parameters["gamut_mode"],
        tone_rolloff=parameters["tone_rolloff"],
        shadow_floor_l=parameters["shadow_floor_l"],
        highlight_ceiling_l=parameters["highlight_ceiling_l"],
        preserve_luma_detail_strength=parameters["preserve_luma_detail"],
        chroma_curve_strength=parameters["chroma_curve_strength"],
        output_margin=parameters["output_margin"],
        guardrails=guardrails,
        dither=parameters["dither"],
    )
    actual = render_style_safe_working_image(
        working,
        profile=profile,
        style=style,
        style_statistics=statistics,
        guardrails=guardrails,
        seed=7,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(working.pixels, frozen)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous


def test_public_engine_rejects_profile_escalation_and_missing_style(working) -> None:
    profile = load_render_profile(PROFILE_PATH, root=ROOT)
    escalated = copy.deepcopy(profile)
    escalated["evidence"]["calibrated_reference_allowed"] = True
    with pytest.raises(ValueError, match="calibrated Reference"):
        render_style_safe_working_image(
            working,
            profile=escalated,
            style="velvia_50",
            style_statistics={},
            guardrails={},
            seed=7,
        )
    with pytest.raises(StyleSafeEngineError, match="absent from profile"):
        render_style_safe_working_image(
            working,
            profile=profile,
            style="not_a_style",
            style_statistics={},
            guardrails={},
            seed=7,
        )


def test_resolved_engine_rejects_nonfinite_source_and_parameter_drift() -> None:
    profile = load_render_profile(PROFILE_PATH, root=ROOT)
    parameters = profile["style_parameters"]["velvia_50"]
    source = np.zeros((2, 3, 3), dtype=np.float32)
    source[0, 0, 0] = np.nan
    with pytest.raises(StyleSafeEngineError, match="finite bounded"):
        render_resolved_safe_lab_rgb(
            source,
            style="velvia_50",
            style_statistics={},
            style_parameters=parameters,
            guardrails={},
            seed=7,
        )
    changed = dict(parameters)
    changed["extra"] = 1.0
    with pytest.raises(StyleSafeEngineError, match="keys drifted"):
        render_resolved_safe_lab_rgb(
            np.zeros((2, 3, 3), dtype=np.float32),
            style="velvia_50",
            style_statistics={},
            style_parameters=changed,
            guardrails={},
            seed=7,
        )


def test_color_only_recipe_replay_is_exact_cli_output(tmp_path: Path) -> None:
    pixels = np.arange(23 * 31 * 3, dtype=np.uint16).reshape(23, 31, 3)
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    replay = tmp_path / "replay.png"
    Image.fromarray((pixels % 256).astype(np.uint8), mode="RGB").save(source)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_film.py"),
            str(source),
            "--style",
            "portra_400",
            "--use-render-profile",
            "--write-recipe",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    recipe = json.loads(output.with_suffix(".recipe.json").read_text(encoding="utf-8"))
    rendered = replay_style_safe_color_recipe(
        recipe, profile_path=PROFILE_PATH, root=ROOT
    )
    save_srgb8(rendered, replay)
    assert replay.read_bytes() == output.read_bytes()

    forged = copy.deepcopy(recipe)
    forged["render"]["color_parameters"]["strength"] += 0.01
    with pytest.raises(StyleSafeEngineError, match="differ from profile"):
        replay_style_safe_color_recipe(forged, profile_path=PROFILE_PATH, root=ROOT)

    absent = copy.deepcopy(recipe)
    absent["render"]["style"] = "not_a_style"
    with pytest.raises(StyleSafeEngineError, match="absent from profile"):
        replay_style_safe_color_recipe(absent, profile_path=PROFILE_PATH, root=ROOT)

    enabled = copy.deepcopy(recipe)
    enabled["render"]["effects"]["grain"]["strength"] = 0.1
    with pytest.raises(StyleSafeEngineError, match="enabled effects"):
        replay_style_safe_color_recipe(enabled, profile_path=PROFILE_PATH, root=ROOT)
