from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.analytic_render_recipe import (
    AnalyticRenderRecipeError,
    validate_analytic_render_recipe,
    verify_analytic_render_recipe_files,
)
from src.inference.analytic_y_chromaticity_profile import (
    load_analytic_y_chromaticity_profile,
    render_analytic_y_chromaticity_profile,
)
from src.preprocess import load_working_image, save_srgb8

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
PROFILE = ROOT / "configs/render_profiles/analytic_y_chromaticity_cb56_v1.json"


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("cb57")
    y, x = np.mgrid[:83, :127]
    pixels = np.stack(
        (
            (x * 7 + y * 3 + 19) % 256,
            (x * 2 + y * 11 + 47) % 256,
            (x * 13 + y * 5 + 101) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    path = directory / "source.png"
    Image.fromarray(pixels, mode="RGB").save(path)
    return path


def _run(source: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "--output", str(output), *extra],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cb57_cli_is_exact_to_direct_runtime_and_repeats(
    tmp_path: Path, source: Path
) -> None:
    runtime = load_analytic_y_chromaticity_profile(PROFILE, root=ROOT)
    direct, direct_facts = render_analytic_y_chromaticity_profile(
        load_working_image(source), runtime, scratch_root=tmp_path
    )
    direct_path = tmp_path / "direct.png"
    save_srgb8(direct, direct_path)
    outputs = [tmp_path / "first.png", tmp_path / "second.png"]
    for output in outputs:
        result = _run(
            source,
            output,
            "--color-engine",
            "analytic-y-chromaticity",
            "--write-metrics",
        )
        assert result.returncode == 0, result.stderr
    assert (
        outputs[0].read_bytes() == outputs[1].read_bytes() == direct_path.read_bytes()
    )
    metrics = json.loads(outputs[0].with_suffix(".metrics.json").read_text())
    profile = metrics["analytic_research_profile"]
    assert profile["profile_id"] == "analytic-y-chromaticity-cb56-v1"
    assert profile["profile_sha256"] == runtime.profile_sha256
    assert profile["product_default"] is False
    assert profile["selector_facts"] == direct_facts


def test_cb57_default_safe_lab_output_remains_exact(
    tmp_path: Path, source: Path
) -> None:
    implicit = tmp_path / "implicit.png"
    explicit = tmp_path / "explicit.png"
    first = _run(source, implicit)
    second = _run(source, explicit, "--color-engine", "safe_lab")
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert implicit.read_bytes() == explicit.read_bytes()


def test_cb57_procedural_effects_repeat_after_opt_in_colour(
    tmp_path: Path, source: Path
) -> None:
    outputs = [tmp_path / "effects-a.png", tmp_path / "effects-b.png"]
    for output in outputs:
        result = _run(
            source,
            output,
            "--color-engine",
            "analytic-y-chromaticity",
            "--grain",
            "0.08",
            "--halation",
            "0.10",
            "--dust",
            "0.05",
            "--seed",
            "23",
        )
        assert result.returncode == 0, result.stderr
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


@pytest.mark.parametrize(
    "extra",
    [
        ("--style", "ektar_100"),
        ("--use-render-profile",),
    ],
)
def test_cb57_unsupported_combinations_fail_before_output(
    tmp_path: Path, source: Path, extra: tuple[str, ...]
) -> None:
    output = tmp_path / ("-".join(value.lstrip("-") for value in extra) + ".png")
    result = _run(source, output, "--color-engine", "analytic-y-chromaticity", *extra)
    assert result.returncode != 0
    assert not output.exists()


def test_cb58_analytic_recipe_verifies_and_binds_files(
    tmp_path: Path, source: Path
) -> None:
    output = tmp_path / "recipe.png"
    result = _run(
        source,
        output,
        "--color-engine",
        "analytic-y-chromaticity",
        "--grain",
        "0.02",
        "--write-recipe",
        "--write-metrics",
    )
    assert result.returncode == 0, result.stderr
    recipe_path = output.with_suffix(".recipe.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    verify_analytic_render_recipe_files(recipe, profile_path=PROFILE, root=ROOT)
    assert recipe["schema_id"] == "kmcfm.analytic-render-recipe.v1"
    assert recipe["render"]["engine_id"] == "analytic_y_chromaticity_cb56_v1"
    assert recipe["claim"]["calibrated_reference_allowed"] is False
    metrics = json.loads(output.with_suffix(".metrics.json").read_text())
    assert metrics["render_recipe"]["schema_id"] == recipe["schema_id"]

    identity_drift = copy.deepcopy(recipe)
    identity_drift["profile"]["profile_id"] = "forged-profile"
    with pytest.raises(AnalyticRenderRecipeError):
        validate_analytic_render_recipe(identity_drift)

    output.write_bytes(output.read_bytes() + b"tamper")
    with pytest.raises(AnalyticRenderRecipeError, match="output file drift"):
        verify_analytic_render_recipe_files(recipe, profile_path=PROFILE, root=ROOT)


def test_cb57_profile_asset_drift_fails_before_output(
    tmp_path: Path, source: Path
) -> None:
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    payload["assets"][0]["sha256"] = "0" * 64
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "drifted.png"
    result = _run(
        source,
        output,
        "--color-engine",
        "analytic-y-chromaticity",
        "--analytic-profile",
        str(drifted),
    )
    assert result.returncode != 0
    assert not output.exists()
