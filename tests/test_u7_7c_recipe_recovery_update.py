from __future__ import annotations

from pathlib import Path

import pytest

import src.inference.recipe_recovery_bundle as recovery
from src.inference import (
    RecipeRecoveryBundleError,
    build_recipe_recovery_bundle,
    inspect_materialized_recipe_recovery_tree,
    materialize_recipe_recovery_bundle,
    update_materialized_recipe_recovery_tree,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
RECIPES = {
    stock: ROOT / f"outputs/eval/u7_2_three_stock_24mp_smoke/{stock}.recipe.json"
    for stock in ("velvia_50", "portra_400", "ektar_100")
}


def _bundle(tmp_path: Path, stock: str) -> Path:
    path = tmp_path / f"{stock}.zip"
    build_recipe_recovery_bundle(
        recipe_path=RECIPES[stock],
        profile_path=PROFILE,
        root=ROOT,
        bundle_path=path,
    )
    return path


@pytest.mark.parametrize(
    ("old_stock", "new_stock"),
    [
        (old_stock, new_stock)
        for old_stock in RECIPES
        for new_stock in RECIPES
        if old_stock != new_stock
    ],
)
def test_updates_all_ordered_stock_pairs_exactly(
    tmp_path: Path, old_stock: str, new_stock: str
) -> None:
    old_bundle = _bundle(tmp_path, old_stock)
    new_bundle = _bundle(tmp_path, new_stock)
    destination = tmp_path / "restored"
    materialize_recipe_recovery_bundle(
        bundle_path=old_bundle, destination_root=destination
    )
    previous = inspect_materialized_recipe_recovery_tree(destination)

    receipt = update_materialized_recipe_recovery_tree(
        bundle_path=new_bundle, destination_root=destination
    )
    updated = inspect_materialized_recipe_recovery_tree(destination)

    assert receipt["previous_style"] == previous["style"] == old_stock
    assert receipt["updated_style"] == updated["style"] == new_stock
    assert receipt["previous_tree_sha256"] == previous["tree_sha256"]
    assert receipt["updated_tree_sha256"] == updated["tree_sha256"]
    assert list(tmp_path.glob(".restored.update-*")) == []
    assert list(tmp_path.glob(".restored.backup-*")) == []


def test_update_rejects_tampered_existing_tree_before_swap(tmp_path: Path) -> None:
    old_bundle = _bundle(tmp_path, "velvia_50")
    new_bundle = _bundle(tmp_path, "portra_400")
    destination = tmp_path / "restored"
    materialize_recipe_recovery_bundle(
        bundle_path=old_bundle, destination_root=destination
    )
    recipe = destination / "recipe.json"
    recipe.write_bytes(recipe.read_bytes() + b" ")

    with pytest.raises(RecipeRecoveryBundleError, match="identity mismatch"):
        update_materialized_recipe_recovery_tree(
            bundle_path=new_bundle, destination_root=destination
        )
    assert recipe.read_bytes().endswith(b" ")
    assert list(tmp_path.glob(".restored.*-*")) == []


def test_post_backup_failure_restores_previous_tree_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_bundle = _bundle(tmp_path, "velvia_50")
    new_bundle = _bundle(tmp_path, "ektar_100")
    destination = tmp_path / "restored"
    materialize_recipe_recovery_bundle(
        bundle_path=old_bundle, destination_root=destination
    )
    before = inspect_materialized_recipe_recovery_tree(destination)

    def fail_publish(stage: Path, destination_root: Path) -> None:
        del stage, destination_root
        raise OSError("injected update publication failure")

    monkeypatch.setattr(recovery, "_publish_recipe_recovery_update", fail_publish)
    with pytest.raises(OSError, match="injected update publication failure"):
        update_materialized_recipe_recovery_tree(
            bundle_path=new_bundle, destination_root=destination
        )

    after = inspect_materialized_recipe_recovery_tree(destination)
    assert after == before
    assert list(tmp_path.glob(".restored.update-*")) == []
    assert list(tmp_path.glob(".restored.backup-*")) == []
