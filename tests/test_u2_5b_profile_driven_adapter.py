from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference import load_render_profile, verify_render_recipe_files


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_film.py"
PROFILE = ROOT / "configs" / "render_profiles" / "safe_rich_v1.json"
CONFIG = json.loads(
    (ROOT / "configs" / "u2_5b_profile_driven_safe_lab_v1.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("u2_5b")
    y, x = np.mgrid[:37, :53]
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
    return path


def _run(source: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "--output", str(output), *extra],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("style", CONFIG["styles_srgb8"])
def test_profile_adapter_is_byte_identical_to_legacy_srgb8(
    tmp_path: Path, source: Path, style: str
) -> None:
    legacy = tmp_path / f"{style}-legacy.png"
    profile = tmp_path / f"{style}-profile.png"
    first = _run(source, legacy, "--style", style)
    second = _run(source, profile, "--style", style, "--use-render-profile")
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert legacy.read_bytes() == profile.read_bytes()


@pytest.mark.parametrize("style", CONFIG["styles_srgb16"])
def test_profile_adapter_is_sample_identical_to_legacy_srgb16(
    tmp_path: Path, source: Path, style: str
) -> None:
    legacy = tmp_path / f"{style}-legacy16.png"
    profile = tmp_path / f"{style}-profile16.png"
    first = _run(source, legacy, "--style", style, "--output-bit-depth", "16")
    second = _run(
        source,
        profile,
        "--style",
        style,
        "--output-bit-depth",
        "16",
        "--use-render-profile",
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    with Image.open(legacy) as left, Image.open(profile) as right:
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_profile_adapter_repeats_and_records_metrics(
    tmp_path: Path, source: Path
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    for output in (first, second):
        result = _run(
            source,
            output,
            "--use-render-profile",
            "--write-metrics",
        )
        assert result.returncode == 0, result.stderr
    assert first.read_bytes() == second.read_bytes()
    first_metrics = json.loads(first.with_suffix(".metrics.json").read_text())
    second_metrics = json.loads(second.with_suffix(".metrics.json").read_text())
    assert first_metrics["profile_driven_adapter"] is True
    assert second_metrics["profile_driven_adapter"] is True


def test_default_metrics_records_legacy_path(tmp_path: Path, source: Path) -> None:
    output = tmp_path / "legacy.png"
    result = _run(source, output, "--write-metrics")
    assert result.returncode == 0, result.stderr
    metrics = json.loads(output.with_suffix(".metrics.json").read_text())
    assert metrics["profile_driven_adapter"] is False


def test_profile_driven_recipe_verifies(tmp_path: Path, source: Path) -> None:
    output = tmp_path / "recipe.png"
    result = _run(
        source,
        output,
        "--style",
        "ektar_100",
        "--use-render-profile",
        "--write-recipe",
    )
    assert result.returncode == 0, result.stderr
    recipe = json.loads(output.with_suffix(".recipe.json").read_text())
    verify_render_recipe_files(recipe, profile_path=PROFILE, root=ROOT)


@pytest.mark.parametrize("failure", ["escalation", "asset", "missing_style"])
def test_invalid_profile_fails_before_any_output(
    tmp_path: Path, source: Path, failure: str
) -> None:
    profile = load_render_profile(PROFILE, root=ROOT)
    if failure == "escalation":
        profile["evidence"]["calibrated_reference_allowed"] = True
    elif failure == "asset":
        profile["assets"][0]["sha256"] = "0" * 64
    else:
        profile["style_parameters"].pop("velvia_50")
    profile_path = tmp_path / f"{failure}.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    output = tmp_path / f"{failure}.png"

    result = _run(
        source,
        output,
        "--render-profile",
        str(profile_path),
        "--use-render-profile",
        "--write-metrics",
        "--write-recipe",
        "--write-layers",
    )
    assert result.returncode != 0
    assert not output.exists()
    assert not output.with_suffix(".metrics.json").exists()
    assert not output.with_suffix(".recipe.json").exists()
    assert not (tmp_path / f"{failure}_layers").exists()


def test_recipe_only_invalid_profile_also_fails_before_output(
    tmp_path: Path, source: Path
) -> None:
    profile = copy.deepcopy(load_render_profile(PROFILE, root=ROOT))
    profile["assets"][1]["sha256"] = "f" * 64
    profile_path = tmp_path / "recipe-invalid.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    output = tmp_path / "recipe-invalid.png"

    result = _run(
        source,
        output,
        "--render-profile",
        str(profile_path),
        "--write-recipe",
    )
    assert result.returncode != 0
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()
