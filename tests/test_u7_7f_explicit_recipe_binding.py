from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.inference.recipe_recovery_bundle as recovery
from src.inference import bind_portable_recipe_recovery_bundle as public_binder
from src.inference.portable_recipe_bundle import (
    bind_portable_recipe_recovery_bundle,
    build_portable_recipe_recovery_bundle,
)
from src.inference.recipe_recovery_bundle import RecipeRecoveryBundleError
from src.inference.render_contract import validate_render_recipe

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
RECIPE = ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke/velvia_50.recipe.json"


def _bundle(tmp_path: Path) -> tuple[Path, dict]:
    source = json.loads(RECIPE.read_text(encoding="utf-8"))
    bundle = tmp_path / "portable.zip"
    build_portable_recipe_recovery_bundle(
        recipe_path=RECIPE,
        profile_path=PROFILE,
        root=ROOT,
        bundle_path=bundle,
    )
    return bundle, source


def test_explicit_binding_is_a_public_inference_callable() -> None:
    assert public_binder is bind_portable_recipe_recovery_bundle


def test_explicit_binding_publishes_strict_recipe_without_output(tmp_path: Path) -> None:
    bundle, source = _bundle(tmp_path)
    output = tmp_path / "future.png"
    destination = tmp_path / "bound.recipe.json"
    result = bind_portable_recipe_recovery_bundle(
        bundle_path=bundle,
        input_path=Path(source["input"]["path"]),
        output_path=output,
        recipe_path=destination,
    )
    bound = json.loads(destination.read_text(encoding="utf-8"))
    validate_render_recipe(bound)
    assert bound["input"]["path"] == source["input"]["path"]
    assert bound["output"]["path"] == str(output)
    source["output"]["path"] = str(output)
    assert bound == source
    assert result["rendered"] is False
    assert result["style"] == "velvia_50"
    assert not output.exists()


def test_explicit_binding_rejects_input_hash_mismatch_before_publish(
    tmp_path: Path,
) -> None:
    bundle, _ = _bundle(tmp_path)
    input_path = tmp_path / "wrong.bin"
    input_path.write_bytes(b"not-the-bound-input")
    destination = tmp_path / "bound.recipe.json"
    with pytest.raises(RecipeRecoveryBundleError, match="input hash"):
        bind_portable_recipe_recovery_bundle(
            bundle_path=bundle,
            input_path=input_path,
            output_path=tmp_path / "future.png",
            recipe_path=destination,
        )
    assert not destination.exists()


@pytest.mark.parametrize("occupied", ["output", "recipe"])
def test_explicit_binding_preserves_existing_destinations(
    tmp_path: Path, occupied: str
) -> None:
    bundle, source = _bundle(tmp_path)
    output = tmp_path / "future.png"
    destination = tmp_path / "bound.recipe.json"
    selected = output if occupied == "output" else destination
    selected.write_bytes(b"foreign")
    with pytest.raises(RecipeRecoveryBundleError, match="already exists"):
        bind_portable_recipe_recovery_bundle(
            bundle_path=bundle,
            input_path=Path(source["input"]["path"]),
            output_path=output,
            recipe_path=destination,
        )
    assert selected.read_bytes() == b"foreign"
    if occupied == "output":
        assert not destination.exists()


def test_explicit_binding_failed_write_removes_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _bundle(tmp_path)
    destination = tmp_path / "bound.recipe.json"
    original_write = recovery.os.write
    injected = False

    def fail_after_partial(descriptor: int, payload: bytes | memoryview) -> int:
        nonlocal injected
        if not injected:
            injected = True
            original_write(descriptor, payload[: max(1, len(payload) // 2)])
            raise OSError("injected partial write")
        return original_write(descriptor, payload)

    monkeypatch.setattr(recovery.os, "write", fail_after_partial)
    with pytest.raises(OSError, match="injected partial write"):
        bind_portable_recipe_recovery_bundle(
            bundle_path=bundle,
            input_path=Path(source["input"]["path"]),
            output_path=tmp_path / "future.png",
            recipe_path=destination,
        )
    assert not destination.exists()
