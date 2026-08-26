#!/usr/bin/env python3
"""Formal U7.7C atomic recovery-tree update audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.inference.recipe_recovery_bundle as recovery
from src.inference import (
    RecipeRecoveryBundleError,
    atomic_write_json,
    build_recipe_recovery_bundle,
    inspect_materialized_recipe_recovery_tree,
    materialize_recipe_recovery_bundle,
    sha256_file,
    update_materialized_recipe_recovery_tree,
)

CONTRACT = ROOT / "configs/u7_7c_recipe_recovery_update_v1.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
RECIPES = {
    stock: ROOT / f"outputs/eval/u7_2_three_stock_24mp_smoke/{stock}.recipe.json"
    for stock in ("velvia_50", "portra_400", "ektar_100")
}
CODE_PATHS = (
    "src/inference/recipe_recovery_bundle.py",
    "src/inference/__init__.py",
    "scripts/audit_u7_7c_recipe_recovery_update.py",
)


def _stable_id(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_bundle(root: Path, stock: str) -> Path:
    path = root / f"{stock}.zip"
    build_recipe_recovery_bundle(
        recipe_path=RECIPES[stock],
        profile_path=PROFILE,
        root=ROOT,
        bundle_path=path,
    )
    return path


def run(*, reverse: bool) -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    code = {relative: sha256_file(ROOT / relative) for relative in CODE_PATHS}
    pairs = [
        (old_stock, new_stock)
        for old_stock in RECIPES
        for new_stock in RECIPES
        if old_stock != new_stock
    ]
    if reverse:
        pairs.reverse()
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="kmcfm-u7-7c-") as raw_root:
        temp_root = Path(raw_root)
        bundles = {
            stock: _build_bundle(temp_root, stock) for stock in sorted(RECIPES)
        }
        bundle_bytes = {stock: path.read_bytes() for stock, path in bundles.items()}
        for index, (old_stock, new_stock) in enumerate(pairs):
            destination = temp_root / f"pair-{index:02d}"
            materialize_recipe_recovery_bundle(
                bundle_path=bundles[old_stock], destination_root=destination
            )
            previous = inspect_materialized_recipe_recovery_tree(destination)
            receipt = update_materialized_recipe_recovery_tree(
                bundle_path=bundles[new_stock], destination_root=destination
            )
            updated = inspect_materialized_recipe_recovery_tree(destination)
            rows.append(
                {
                    "old_style": old_stock,
                    "new_style": new_stock,
                    "old_tree_sha256": previous["tree_sha256"],
                    "new_tree_sha256": updated["tree_sha256"],
                    "receipt_sha256": _stable_id(receipt),
                    "receipt_binds_trees": receipt["previous_tree_sha256"]
                    == previous["tree_sha256"]
                    and receipt["updated_tree_sha256"] == updated["tree_sha256"],
                    "member_count": updated["member_count"],
                }
            )

        fault_root = temp_root / "fault"
        materialize_recipe_recovery_bundle(
            bundle_path=bundles["velvia_50"], destination_root=fault_root
        )
        fault_before = inspect_materialized_recipe_recovery_tree(fault_root)
        original_publish = recovery._publish_recipe_recovery_update

        def fail_publish(stage: Path, destination_root: Path) -> None:
            del stage, destination_root
            raise OSError("injected post-backup publication failure")

        recovery._publish_recipe_recovery_update = fail_publish
        injected_failure_observed = False
        try:
            update_materialized_recipe_recovery_tree(
                bundle_path=bundles["portra_400"], destination_root=fault_root
            )
        except OSError as exc:
            injected_failure_observed = str(exc) == (
                "injected post-backup publication failure"
            )
        finally:
            recovery._publish_recipe_recovery_update = original_publish
        fault_after = inspect_materialized_recipe_recovery_tree(fault_root)

        tamper_root = temp_root / "tamper"
        materialize_recipe_recovery_bundle(
            bundle_path=bundles["ektar_100"], destination_root=tamper_root
        )
        tampered_recipe = tamper_root / "recipe.json"
        tampered_bytes = tampered_recipe.read_bytes() + b" "
        tampered_recipe.write_bytes(tampered_bytes)
        tamper_rejected = False
        try:
            update_materialized_recipe_recovery_tree(
                bundle_path=bundles["velvia_50"], destination_root=tamper_root
            )
        except RecipeRecoveryBundleError:
            tamper_rejected = True
        tampered_existing_tree_unchanged = (
            tampered_recipe.read_bytes() == tampered_bytes
        )

        residue = sorted(
            path.name
            for path in temp_root.iterdir()
            if path.name.startswith(".")
            and (".update-" in path.name or ".backup-" in path.name)
        )
        bundles_unchanged = all(
            bundles[stock].read_bytes() == payload
            for stock, payload in bundle_bytes.items()
        )

    rows.sort(key=lambda row: (row["old_style"], row["new_style"]))
    expected_pairs = sorted(
        (old_stock, new_stock)
        for old_stock in RECIPES
        for new_stock in RECIPES
        if old_stock != new_stock
    )
    gates = {
        "all_six_ordered_style_pairs": [
            (row["old_style"], row["new_style"]) for row in rows
        ]
        == expected_pairs,
        "all_receipts_bind_exact_trees": all(
            row["receipt_binds_trees"] for row in rows
        ),
        "all_updated_trees_have_six_members": all(
            row["member_count"] == 6 for row in rows
        ),
        "post_backup_failure_observed": injected_failure_observed,
        "post_backup_failure_restores_old_tree": fault_after == fault_before,
        "tampered_existing_tree_rejected": tamper_rejected,
        "tampered_existing_tree_unchanged": tampered_existing_tree_unchanged,
        "source_bundles_unchanged": bundles_unchanged,
        "owned_update_residue_zero": residue == [],
    }
    scientific = {
        "schema": "kmcfm.u7-7c-recipe-recovery-update-result.v1",
        "node": "U7.7C",
        "contract_sha256": sha256_file(CONTRACT),
        "implementation_sha256": code,
        "rows": rows,
        "gates": gates,
        "decision": (
            "PASS_PRIVATE_ATOMIC_RECIPE_RECOVERY_UPDATE"
            if all(gates.values())
            else "FAIL_CLOSED_RECIPE_RECOVERY_UPDATE"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    scientific["stable_id"] = _stable_id(scientific)
    return scientific


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(reverse=args.reverse)
    atomic_write_json(args.output, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
