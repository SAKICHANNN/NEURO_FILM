from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import src.inference.recipe_recovery_bundle as recovery
from src.inference import (
    RecipeRecoveryBundleError,
    build_recipe_recovery_bundle,
    inspect_recipe_recovery_bundle,
    materialize_recipe_recovery_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
RECIPE = ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke/velvia_50.recipe.json"


def _build(path: Path) -> dict:
    return build_recipe_recovery_bundle(
        recipe_path=RECIPE,
        profile_path=PROFILE,
        root=ROOT,
        bundle_path=path,
    )


def test_materialize_restores_exact_validated_tree(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    _build(bundle)
    destination = tmp_path / "restored"
    first = materialize_recipe_recovery_bundle(
        bundle_path=bundle, destination_root=destination
    )
    assert first["member_count"] == 6
    assert first["privacy"]["includes_input_bytes"] is False
    assert sorted(
        path.relative_to(destination).as_posix() for path in destination.rglob("*")
    ) == [
        "manifest.json",
        "payload",
        "payload/configs",
        "payload/configs/color_guardrails.json",
        "payload/configs/color_rendering_profiles.yaml",
        "payload/configs/film_color_stats.json",
        "payload/configs/render_profiles",
        "payload/configs/render_profiles/safe_rich_v1.json",
        "recipe.json",
    ]
    with pytest.raises(RecipeRecoveryBundleError, match="already exists"):
        materialize_recipe_recovery_bundle(
            bundle_path=bundle, destination_root=destination
        )


def test_materialize_failure_rolls_back_owned_stage(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = tmp_path / "bundle.zip"
    _build(bundle)
    destination = tmp_path / "restored"
    original = recovery._write_restored_member
    calls = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected restore failure")
        original(path, payload)

    monkeypatch.setattr(recovery, "_write_restored_member", fail_second)
    with pytest.raises(OSError, match="injected"):
        materialize_recipe_recovery_bundle(
            bundle_path=bundle, destination_root=destination
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".restored.restore-*")) == []


def test_cli_restore_receipt_matches_api(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    _build(bundle)
    destination = tmp_path / "restored"
    restored = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/package_film_recipe.py"),
            "restore",
            "--bundle",
            str(bundle),
            "--destination",
            str(destination),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert restored.returncode == 0, restored.stderr
    assert (
        json.loads(restored.stdout)["bundle_sha256"]
        == inspect_recipe_recovery_bundle(bundle)["bundle_sha256"]
    )
