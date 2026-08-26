from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.recipe_history_export import (
    RecipeHistoryExportError,
    export_recipe_history_entry,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"


@pytest.fixture()
def rendered_history(tmp_path: Path) -> dict[str, Path | bytes | str]:
    source = tmp_path / "source.png"
    prior_output = tmp_path / "prior.png"
    history = tmp_path / "history"
    history.mkdir()
    pixels = np.arange(29 * 37 * 3, dtype=np.uint32).reshape(29, 37, 3)
    Image.fromarray((pixels % 256).astype(np.uint8), mode="RGB").save(source)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source),
            "--style",
            "ektar_100",
            "--use-render-profile",
            "--output-bit-depth",
            "16",
            "--write-recipe",
            "--output",
            str(prior_output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    recipe_source = prior_output.with_suffix(".recipe.json")
    recipe_path = history / "ektar_100.recipe.json"
    recipe_path.write_bytes(recipe_source.read_bytes())
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    return {
        "history": history,
        "recipe_path": recipe_path,
        "prior_output": prior_output,
        "expected": prior_output.read_bytes(),
        "expected_sha256": recipe["output"]["sha256"],
    }


def test_history_selection_replays_exact_output_without_prior_output_read(
    rendered_history: dict[str, Path | bytes | str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_output = rendered_history["prior_output"]
    assert isinstance(prior_output, Path)
    original_open = Path.open
    opened: list[Path] = []

    def tracked_open(path: Path, *args: object, **kwargs: object):
        opened.append(path.resolve())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    destination = tmp_path / "export.png"
    receipt = export_recipe_history_entry(
        rendered_history["history"],  # type: ignore[arg-type]
        recipe_path="ektar_100.recipe.json",
        profile_path=PROFILE,
        output_path=destination,
        root=ROOT,
    )

    assert destination.read_bytes() == rendered_history["expected"]
    assert receipt["output_sha256"] == rendered_history["expected_sha256"]
    assert receipt["style"] == "ektar_100"
    assert prior_output.resolve() not in opened


def test_missing_selection_and_existing_destination_fail_before_render(
    rendered_history: dict[str, Path | bytes | str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_replay(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("render must not start")

    monkeypatch.setattr(
        "src.inference.recipe_history_export.replay_style_safe_recipe_to_file",
        forbidden_replay,
    )
    with pytest.raises(RecipeHistoryExportError, match="missing or invalid"):
        export_recipe_history_entry(
            rendered_history["history"],  # type: ignore[arg-type]
            recipe_path="missing.recipe.json",
            profile_path=PROFILE,
            output_path=tmp_path / "missing.png",
            root=ROOT,
        )
    existing = tmp_path / "existing.png"
    existing.write_bytes(b"owned")
    with pytest.raises(RecipeHistoryExportError, match="already exists"):
        export_recipe_history_entry(
            rendered_history["history"],  # type: ignore[arg-type]
            recipe_path="ektar_100.recipe.json",
            profile_path=PROFILE,
            output_path=existing,
            root=ROOT,
        )
    assert existing.read_bytes() == b"owned"
    assert calls == 0


def test_recipe_change_after_catalog_validation_fails_before_render(
    rendered_history: dict[str, Path | bytes | str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import recipe_history_export as module

    original_build = module.build_render_recipe_history
    recipe_path = rendered_history["recipe_path"]
    assert isinstance(recipe_path, Path)

    def build_then_change(*args: object, **kwargs: object) -> dict:
        result = original_build(*args, **kwargs)
        recipe_path.write_bytes(recipe_path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(module, "build_render_recipe_history", build_then_change)
    with pytest.raises(RecipeHistoryExportError, match="changed"):
        export_recipe_history_entry(
            rendered_history["history"],  # type: ignore[arg-type]
            recipe_path="ektar_100.recipe.json",
            profile_path=PROFILE,
            output_path=tmp_path / "changed.png",
            root=ROOT,
        )
    assert not (tmp_path / "changed.png").exists()


def test_history_export_cli_emits_exact_receipt(
    rendered_history: dict[str, Path | bytes | str], tmp_path: Path
) -> None:
    destination = tmp_path / "cli.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_recipe_history.py"),
            "--history-root",
            str(rendered_history["history"]),
            "--recipe-path",
            "ektar_100.recipe.json",
            "--output",
            str(destination),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["output_sha256"] == rendered_history["expected_sha256"]
    assert destination.read_bytes() == rendered_history["expected"]
