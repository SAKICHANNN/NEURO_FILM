from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
README = ROOT / "README.md"


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


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _normalized_recipe(path: Path) -> bytes:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["software"]["commit"] = "<COMMIT>"
    payload["input"]["path"] = "<INPUT>"
    payload["output"]["path"] = "<OUTPUT>"
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize("look", ["velvia_50", "portra_400", "ektar_100"])
@pytest.mark.parametrize("amount", [0.0, 0.5, 1.0])
def test_product_look_entry_matches_existing_explicit_product_path(
    tmp_path: Path, look: str, amount: float
) -> None:
    source = tmp_path / "source.png"
    reference = tmp_path / f"reference-{look}-{amount}.png"
    candidate = tmp_path / f"candidate-{look}-{amount}.png"
    _source(source)
    common = (
        str(source),
        "--look-amount",
        str(amount),
        "--write-recipe",
    )
    first = _run(
        *common,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--style",
        look,
        "--output",
        str(reference),
    )
    second = _run(
        *common,
        "--product-look",
        look,
        "--output",
        str(candidate),
    )
    assert first.returncode == second.returncode == 0, (first.stderr, second.stderr)
    assert reference.read_bytes() == candidate.read_bytes()
    assert _normalized_recipe(reference.with_suffix(".recipe.json")) == (
        _normalized_recipe(candidate.with_suffix(".recipe.json"))
    )
    recipe = json.loads(candidate.with_suffix(".recipe.json").read_text("utf-8"))
    assert recipe["profile"]["profile_id"] == "safe-rich-product-v1"
    assert recipe["render"]["style"] == look
    assert recipe["claim"]["output_label"] == "film-inspired"
    assert recipe["claim"]["calibrated_reference_allowed"] is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--style", "portra_400"), "cannot be combined with --style"),
        (("--use-render-profile",), "cannot be combined with --use-render-profile"),
        (
            ("--render-profile", str(PRODUCT_PROFILE)),
            "cannot be combined with --render-profile",
        ),
        (
            ("--color-engine", "analytic-y-chromaticity"),
            "requires the safe_lab color engine",
        ),
    ],
)
def test_product_look_conflicts_reject_before_input_decode(
    tmp_path: Path, arguments: tuple[str, ...], message: str
) -> None:
    output = tmp_path / "must-not-exist.png"
    completed = _run(
        str(tmp_path / "must-not-decode.png"),
        "--product-look",
        "ektar_100",
        *arguments,
        "--output",
        str(output),
    )
    assert completed.returncode == 2
    assert message in completed.stderr
    assert "must-not-decode" not in completed.stderr
    assert not output.exists()


def test_product_look_rejects_discovery_and_non_product_values(tmp_path: Path) -> None:
    conflict = _run("--product-look", "ektar_100", "--list-product-looks")
    assert conflict.returncode == 2
    assert "cannot be combined with --list-product-looks" in conflict.stderr

    for value in ("generic_bw", "hp5", "unknown", ""):
        output = tmp_path / f"{value or 'empty'}.png"
        completed = _run(
            str(tmp_path / "must-not-decode.png"),
            "--product-look",
            value,
            "--output",
            str(output),
        )
        assert completed.returncode == 2
        assert "invalid choice" in completed.stderr
        assert not output.exists()


def test_readme_primary_command_and_claim_boundary_are_current() -> None:
    text = README.read_text(encoding="utf-8")
    assert "--product-look portra_400" in text
    assert "--look-amount 0.75" in text
    assert "--write-recipe" in text
    assert "not claims of calibrated stock response or physical-film" in text
    assert "--style portra_400" not in text


def test_product_look_full_auxiliary_bundle_matches_explicit_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    _source(source)
    common = (
        str(source),
        "--look-amount",
        "0.5",
        "--grain",
        "0.05",
        "--halation",
        "0.15",
        "--dust",
        "0.02",
        "--write-recipe",
        "--write-layers",
        "--write-metrics",
    )
    first = _run(
        *common,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--style",
        "ektar_100",
        "--output",
        str(reference),
    )
    second = _run(
        *common,
        "--product-look",
        "ektar_100",
        "--output",
        str(candidate),
    )
    assert first.returncode == second.returncode == 0, (first.stderr, second.stderr)
    assert reference.read_bytes() == candidate.read_bytes()
    assert _normalized_recipe(reference.with_suffix(".recipe.json")) == (
        _normalized_recipe(candidate.with_suffix(".recipe.json"))
    )
    for suffix in (".metrics.json",):
        left = json.loads(reference.with_suffix(suffix).read_text("utf-8"))
        right = json.loads(candidate.with_suffix(suffix).read_text("utf-8"))
        for payload in (left, right):
            payload["input"] = "<INPUT>"
            payload["output"] = "<OUTPUT>"
            payload["render_recipe"]["path"] = "<RECIPE>"
            payload["render_recipe"]["sha256"] = "<RECIPE_SHA>"
        assert left == right
    left_layers = reference.parent / f"{reference.stem}_layers"
    right_layers = candidate.parent / f"{candidate.stem}_layers"
    assert [path.name for path in left_layers.iterdir()] == [
        path.name for path in right_layers.iterdir()
    ]
    for left in left_layers.iterdir():
        right = right_layers / left.name
        assert (
            hashlib.sha256(left.read_bytes()).digest()
            == hashlib.sha256(right.read_bytes()).digest()
        )
