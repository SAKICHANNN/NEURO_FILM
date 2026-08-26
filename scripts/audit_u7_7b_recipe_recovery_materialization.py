#!/usr/bin/env python3
"""Formal U7.7B create-only recovery materialization audit."""

# ruff: noqa: I001

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

from src.inference import (
    RecipeRecoveryBundleError,
    atomic_write_json,
    build_recipe_recovery_bundle,
    materialize_recipe_recovery_bundle,
    sha256_file,
)


CONTRACT = ROOT / "configs/u7_7b_recipe_recovery_materialization_v1.json"
PARENT_CONTRACT = ROOT / "configs/u7_7a_recipe_recovery_bundle_v1.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
IMPLEMENTATION_COMMIT = "f22deeab5c6ca9e516e7017413d3ac7bff5a5b2b"
CODE = {
    "src/inference/recipe_recovery_bundle.py": "6c4d38ab8d9b89acfe6d4216737e7a0a3d7488ebfe8e7014c8409fb9edc4decb",
    "scripts/package_film_recipe.py": "a619c4054a415d5b348c03db3290b49a3adee4b88da6462ae5e256dc62a1d642",
}
CONTRACT_SHA256 = "8b5df36063b65071ba1c76c6b21dd3c89860b19449ba7a936bf172b87b5d0434"


def _stable_id(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run(*, reverse: bool) -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_CONTRACT.read_text(encoding="utf-8"))
    for relative, expected in CODE.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"U7.7B implementation drift: {relative}")
    recipes = list(parent["inputs"]["recipes"])
    if reverse:
        recipes.reverse()
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="kmcfm-u7-7b-") as raw_root:
        temp_root = Path(raw_root)
        for index, relative in enumerate(recipes):
            bundle = temp_root / f"{index:02d}.zip"
            build = build_recipe_recovery_bundle(
                recipe_path=ROOT / relative,
                profile_path=PROFILE,
                root=ROOT,
                bundle_path=bundle,
            )
            bundle_before = bundle.read_bytes()
            first_root = temp_root / f"{index:02d}-restore-a"
            second_root = temp_root / f"{index:02d}-restore-b"
            first = materialize_recipe_recovery_bundle(
                bundle_path=bundle, destination_root=first_root
            )
            second = materialize_recipe_recovery_bundle(
                bundle_path=bundle, destination_root=second_root
            )
            sentinel_root = temp_root / f"{index:02d}-foreign"
            sentinel_root.mkdir()
            sentinel = sentinel_root / "owner.bin"
            sentinel.write_bytes(b"foreign-owner")
            existing_unchanged = False
            try:
                materialize_recipe_recovery_bundle(
                    bundle_path=bundle, destination_root=sentinel_root
                )
            except RecipeRecoveryBundleError:
                existing_unchanged = sentinel.read_bytes() == b"foreign-owner"
            restored_files = sorted(
                path.relative_to(first_root).as_posix()
                for path in first_root.rglob("*")
                if path.is_file()
            )
            rows.append(
                {
                    "style": build["style"],
                    "bundle_sha256": build["bundle_sha256"],
                    "receipt_sha256": _stable_id(first),
                    "repeat_receipt_exact": first == second,
                    "restored_member_count": first["member_count"],
                    "restored_files": restored_files,
                    "bundle_unchanged": bundle.read_bytes() == bundle_before,
                    "existing_destination_unchanged": existing_unchanged,
                    "privacy": first["privacy"],
                    "claim": first["claim"],
                }
            )
        owned_stage_residue = list(temp_root.rglob(".*.restore-*"))
    rows.sort(key=lambda row: row["style"])
    gates = {
        "contract_exact": sha256_file(CONTRACT) == CONTRACT_SHA256,
        "implementation_exact": True,
        "three_styles_complete": [row["style"] for row in rows]
        == ["ektar_100", "portra_400", "velvia_50"],
        "repeat_receipts_exact": all(row["repeat_receipt_exact"] for row in rows),
        "restored_members_exact": all(
            row["restored_member_count"] == 6 for row in rows
        ),
        "bundle_sources_unchanged": all(row["bundle_unchanged"] for row in rows),
        "existing_destinations_unchanged": all(
            row["existing_destination_unchanged"] for row in rows
        ),
        "no_pixel_payloads": all(
            row["privacy"]
            == {
                "includes_input_bytes": False,
                "includes_output_bytes": False,
                "network_required": False,
                "telemetry": False,
            }
            for row in rows
        ),
        "claims_remain_look_approximation": all(
            row["claim"]
            == {
                "calibrated_reference_allowed": False,
                "evidence_grade": "look-approximation",
                "output_label": "film-inspired",
            }
            for row in rows
        ),
        "owned_stage_residue_zero": not owned_stage_residue,
    }
    scientific = {
        "schema": "kmcfm.u7-7b-recipe-recovery-materialization-result.v1",
        "node": "U7.7B",
        "contract_sha256": sha256_file(CONTRACT),
        "parent_contract_sha256": sha256_file(PARENT_CONTRACT),
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "implementation_sha256": CODE,
        "rows": rows,
        "gates": gates,
        "decision": (
            "PASS_PRIVATE_CREATE_ONLY_RECOVERY_MATERIALIZATION"
            if all(gates.values())
            else "FAIL_CLOSED_RECOVERY_MATERIALIZATION"
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
