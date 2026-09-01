from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.recipe_browser_export import (
    RecipeBrowserExportError,
    RecipeBrowserExportSession,
)
from src.inference.recipe_export_request import build_recipe_export_request_set
from src.inference.recipe_history import build_render_recipe_history
from src.inference.recipe_history_export import (
    RecipeHistoryExportError,
    export_recipe_history_entry,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"


@pytest.fixture()
def histories(tmp_path: Path) -> dict[str, Path | bytes]:
    source = tmp_path / "source.png"
    yy, xx = np.mgrid[:192, :256]
    pixels = np.stack(
        (
            (3 * xx + yy) % 256,
            (xx + 5 * yy) % 256,
            ((xx // 8) * 19 + (yy // 6) * 13) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(source)

    research_output = tmp_path / "research.png"
    product_output = tmp_path / "product.png"
    commands = (
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source),
            "--style",
            "velvia_50",
            "--use-render-profile",
            "--tile-size",
            "64",
            "--output-bit-depth",
            "16",
            "--png-compression",
            "0",
            "--write-recipe",
            "--output",
            str(research_output),
            "--halation-model",
            "staged-density-research",
            "--halation",
            "1.0",
        ],
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source),
            "--style",
            "ektar_100",
            "--use-render-profile",
            "--output-bit-depth",
            "16",
            "--png-compression",
            "0",
            "--write-recipe",
            "--output",
            str(product_output),
        ],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    research_only = tmp_path / "research-only"
    product_only = tmp_path / "product-only"
    mixed = tmp_path / "mixed"
    for directory in (research_only, product_only, mixed):
        directory.mkdir()
    research_recipe = research_output.with_suffix(".recipe.json").read_bytes()
    product_recipe = product_output.with_suffix(".recipe.json").read_bytes()
    (research_only / "research.recipe.json").write_bytes(research_recipe)
    (product_only / "product.recipe.json").write_bytes(product_recipe)
    (mixed / "research.recipe.json").write_bytes(research_recipe)
    (mixed / "product.recipe.json").write_bytes(product_recipe)
    return {
        "research_only": research_only,
        "product_only": product_only,
        "mixed": mixed,
        "product_bytes": product_output.read_bytes(),
    }


def test_default_history_retains_research_evidence_but_product_view_rejects(
    histories: dict[str, Path | bytes],
) -> None:
    root = histories["research_only"]
    assert isinstance(root, Path)
    general = build_render_recipe_history(root)
    product = build_render_recipe_history(root, product_export_only=True)
    assert general["status"] == "ready"
    assert general["counts"] == {"discovered": 1, "valid": 1, "invalid": 0}
    assert general["entries"][0]["enabled_effects"] == ["halation"]
    assert product["status"] == "invalid"
    assert product["counts"] == {"discovered": 1, "valid": 0, "invalid": 1}
    assert product["entries"][0]["error_code"] == "recipe_research_only"


def test_product_export_paths_reject_research_recipe_before_replay(
    histories: dict[str, Path | bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = histories["research_only"]
    assert isinstance(root, Path)
    request_set = build_recipe_export_request_set(root)
    assert request_set["receipt"]["request_count"] == 0
    assert request_set["receipt"]["requests"] == []

    monkeypatch.setattr(
        "src.inference.recipe_history_export.replay_style_safe_recipe_to_file",
        lambda *_args, **_kwargs: pytest.fail("research replay must not start"),
    )
    destination = tmp_path / "forbidden.png"
    with pytest.raises(RecipeHistoryExportError, match="missing or invalid"):
        export_recipe_history_entry(
            root,
            recipe_path="research.recipe.json",
            profile_path=PROFILE,
            output_path=destination,
            root=ROOT,
        )
    assert not destination.exists()

    browser_output = tmp_path / "browser-output"
    with pytest.raises(RecipeBrowserExportError, match="no valid export recipes"):
        RecipeBrowserExportSession(
            root,
            browser_output,
            profile_path=PROFILE,
            root=ROOT,
        )
    assert list(browser_output.iterdir()) == []


def test_mixed_history_exports_only_ordinary_product_recipe(
    histories: dict[str, Path | bytes], tmp_path: Path
) -> None:
    mixed = histories["mixed"]
    product_only = histories["product_only"]
    expected = histories["product_bytes"]
    assert isinstance(mixed, Path)
    assert isinstance(product_only, Path)
    assert isinstance(expected, bytes)

    mixed_set = build_recipe_export_request_set(mixed)
    product_set = build_recipe_export_request_set(product_only)
    assert mixed_set["receipt"]["request_count"] == 1
    assert mixed_set["receipt"]["requests"][0]["style"] == "ektar_100"
    assert mixed_set["files"]["index.html"] == product_set["files"]["index.html"]

    destination = tmp_path / "export.png"
    receipt = export_recipe_history_entry(
        mixed,
        recipe_path="product.recipe.json",
        profile_path=PROFILE,
        output_path=destination,
        root=ROOT,
    )
    assert destination.read_bytes() == expected
    assert receipt["style"] == "ektar_100"


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_product_export_only_requires_a_boolean(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="product_export_only must be boolean"):
        build_render_recipe_history(tmp_path, product_export_only=value)  # type: ignore[arg-type]
