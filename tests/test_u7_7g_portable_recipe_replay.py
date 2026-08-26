from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference import replay_portable_recipe_recovery_bundle_to_file
from src.inference.portable_recipe_bundle import (
    build_portable_recipe_recovery_bundle,
)
from src.inference.recipe_recovery_bundle import RecipeRecoveryBundleError
from src.inference.style_safe_engine import StyleSafeEngineError

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"


@pytest.fixture()
def portable_fixture(tmp_path: Path) -> dict[str, Path | bytes]:
    source = tmp_path / "source.png"
    original_output = tmp_path / "original.png"
    pixels = np.arange(31 * 43 * 3, dtype=np.uint32).reshape(31, 43, 3)
    Image.fromarray((pixels % 256).astype(np.uint8), mode="RGB").save(source)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source),
            "--style",
            "portra_400",
            "--use-render-profile",
            "--output-bit-depth",
            "16",
            "--write-recipe",
            "--output",
            str(original_output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    source_recipe = original_output.with_suffix(".recipe.json")
    bundle = tmp_path / "portable.zip"
    build_portable_recipe_recovery_bundle(
        recipe_path=source_recipe,
        profile_path=PROFILE,
        root=ROOT,
        bundle_path=bundle,
    )
    return {
        "source": source,
        "original_output": original_output,
        "expected": original_output.read_bytes(),
        "source_recipe": source_recipe,
        "bundle": bundle,
    }


def test_portable_recipe_replay_regenerates_exact_output(
    portable_fixture: dict[str, Path | bytes], tmp_path: Path
) -> None:
    output = tmp_path / "replayed.png"
    bound_recipe = tmp_path / "bound.recipe.json"
    receipt = replay_portable_recipe_recovery_bundle_to_file(
        bundle_path=portable_fixture["bundle"],  # type: ignore[arg-type]
        input_path=portable_fixture["source"],  # type: ignore[arg-type]
        output_path=output,
        recipe_path=bound_recipe,
        profile_path=PROFILE,
        root=ROOT,
    )
    assert output.read_bytes() == portable_fixture["expected"]
    assert receipt["output_sha256"] == json.loads(
        portable_fixture["source_recipe"].read_text(encoding="utf-8")  # type: ignore[union-attr]
    )["output"]["sha256"]
    assert receipt["style"] == "portra_400"
    assert receipt["caller_driven"] is True
    assert receipt["claim"]["evidence_grade"] == "look-approximation"


def test_portable_recipe_replay_preserves_existing_output_before_binding(
    portable_fixture: dict[str, Path | bytes], tmp_path: Path
) -> None:
    output = tmp_path / "existing.png"
    output.write_bytes(b"foreign")
    bound_recipe = tmp_path / "bound.recipe.json"
    with pytest.raises(RecipeRecoveryBundleError, match="already exists"):
        replay_portable_recipe_recovery_bundle_to_file(
            bundle_path=portable_fixture["bundle"],  # type: ignore[arg-type]
            input_path=portable_fixture["source"],  # type: ignore[arg-type]
            output_path=output,
            recipe_path=bound_recipe,
            profile_path=PROFILE,
            root=ROOT,
        )
    assert output.read_bytes() == b"foreign"
    assert not bound_recipe.exists()


def test_portable_recipe_replay_removes_output_on_identity_failure(
    portable_fixture: dict[str, Path | bytes], tmp_path: Path
) -> None:
    source_recipe = portable_fixture["source_recipe"]
    assert isinstance(source_recipe, Path)
    forged = json.loads(source_recipe.read_text(encoding="utf-8"))
    forged["output"]["sha256"] = "0" * 64
    forged_path = tmp_path / "forged.recipe.json"
    forged_path.write_text(
        json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    bundle = tmp_path / "forged.zip"
    build_portable_recipe_recovery_bundle(
        recipe_path=forged_path,
        profile_path=PROFILE,
        root=ROOT,
        bundle_path=bundle,
    )
    output = tmp_path / "failed.png"
    bound_recipe = tmp_path / "failed-bound.recipe.json"
    with pytest.raises(StyleSafeEngineError, match="byte identity"):
        replay_portable_recipe_recovery_bundle_to_file(
            bundle_path=bundle,
            input_path=portable_fixture["source"],  # type: ignore[arg-type]
            output_path=output,
            recipe_path=bound_recipe,
            profile_path=PROFILE,
            root=ROOT,
        )
    assert not output.exists()
    assert bound_recipe.is_file()
