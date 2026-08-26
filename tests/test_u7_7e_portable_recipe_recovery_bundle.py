from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.inference.portable_recipe_bundle import (
    PORTABLE_BUNDLE_SCHEMA_ID,
    PORTABLE_RECIPE_SCHEMA_ID,
    bind_portable_recipe,
    build_portable_recipe_recovery_bundle,
    inspect_portable_recipe_recovery_bundle,
    portable_recipe_from_strict,
)
from src.inference.recipe_recovery_bundle import RecipeRecoveryBundleError

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
RECIPE = ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke/velvia_50.recipe.json"


def _recipe() -> dict:
    return json.loads(RECIPE.read_text(encoding="utf-8"))


def test_portable_recipe_removes_only_paths_and_rebinds_exactly() -> None:
    original = _recipe()
    portable = portable_recipe_from_strict(original)
    assert portable["schema_id"] == PORTABLE_RECIPE_SCHEMA_ID
    assert "path" not in portable["input"] and "path" not in portable["output"]
    assert portable["input"]["path_binding"] == "unresolved-user-input"
    assert portable["output"]["path_binding"] == "unresolved-user-output"
    rebound = bind_portable_recipe(
        portable,
        input_path=original["input"]["path"],
        output_path=original["output"]["path"],
    )
    assert rebound == original


def test_portable_recipe_accepts_new_paths_without_other_semantic_drift() -> None:
    original = _recipe()
    portable = portable_recipe_from_strict(_recipe())
    rebound = bind_portable_recipe(
        portable, input_path="different-input", output_path="different-output"
    )
    original["input"]["path"] = "different-input"
    original["output"]["path"] = "different-output"
    assert rebound == original


def test_portable_bundle_is_repeat_exact_and_contains_no_absolute_paths(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    one = build_portable_recipe_recovery_bundle(
        recipe_path=RECIPE, profile_path=PROFILE, root=ROOT, bundle_path=first
    )
    two = build_portable_recipe_recovery_bundle(
        recipe_path=RECIPE, profile_path=PROFILE, root=ROOT, bundle_path=second
    )
    assert first.read_bytes() == second.read_bytes()
    assert one == two == inspect_portable_recipe_recovery_bundle(first)
    assert one["schema_id"] == PORTABLE_BUNDLE_SCHEMA_ID
    with zipfile.ZipFile(first) as archive:
        portable = archive.read("portable_recipe.json").decode("utf-8")
    assert "neuro_film_storage" not in portable.casefold()
    parsed = json.loads(portable)
    assert "path" not in parsed["input"] and "path" not in parsed["output"]
    assert "path_binding" in parsed["input"] and "path_binding" in parsed["output"]


def test_portable_bundle_is_create_only_and_tamper_rejects(tmp_path: Path) -> None:
    destination = tmp_path / "bundle.zip"
    destination.write_bytes(b"foreign")
    with pytest.raises(RecipeRecoveryBundleError, match="already exists"):
        build_portable_recipe_recovery_bundle(
            recipe_path=RECIPE,
            profile_path=PROFILE,
            root=ROOT,
            bundle_path=destination,
        )
    assert destination.read_bytes() == b"foreign"

    valid = tmp_path / "valid.zip"
    build_portable_recipe_recovery_bundle(
        recipe_path=RECIPE, profile_path=PROFILE, root=ROOT, bundle_path=valid
    )
    payload = bytearray(valid.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    tampered = tmp_path / "tampered.zip"
    tampered.write_bytes(payload)
    with pytest.raises((RecipeRecoveryBundleError, zipfile.BadZipFile)):
        inspect_portable_recipe_recovery_bundle(tampered)
