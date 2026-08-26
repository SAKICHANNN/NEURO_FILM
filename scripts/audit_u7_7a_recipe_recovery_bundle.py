#!/usr/bin/env python3
"""Formal U7.7A no-pixel recipe recovery-bundle audit."""

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
    inspect_recipe_recovery_bundle,
    sha256_file,
)


CONTRACT = ROOT / "configs/u7_7a_recipe_recovery_bundle_v1.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
IMPLEMENTATION_COMMIT = "38e92c5bb1069d25af2d50e0f5eeacee5ea96cd2"
CODE = {
    "src/inference/recipe_recovery_bundle.py": "18a23cecc04f02eb0ce5bcae74c5916ee7a8e8ac36fb409d758b4418aafeb8df",
    "scripts/package_film_recipe.py": "efe4756d362701223584d016758d681a7f800b6183d507316d230c558d46e191",
}


def _stable_id(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run(*, reverse: bool) -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for relative, expected in CODE.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"U7.7A implementation drift: {relative}")
    recipes = list(contract["inputs"]["recipes"])
    if reverse:
        recipes.reverse()
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="kmcfm-u7-7a-") as raw_root:
        temp_root = Path(raw_root)
        for index, relative in enumerate(recipes):
            recipe_path = ROOT / relative
            first = temp_root / f"{index:02d}-first.zip"
            second = temp_root / f"{index:02d}-second.zip"
            first_result = build_recipe_recovery_bundle(
                recipe_path=recipe_path,
                profile_path=PROFILE,
                root=ROOT,
                bundle_path=first,
            )
            second_result = build_recipe_recovery_bundle(
                recipe_path=recipe_path,
                profile_path=PROFILE,
                root=ROOT,
                bundle_path=second,
            )
            inspected = inspect_recipe_recovery_bundle(first)
            foreign = temp_root / f"{index:02d}-foreign.zip"
            foreign.write_bytes(b"foreign-owner")
            existing_unchanged = False
            try:
                build_recipe_recovery_bundle(
                    recipe_path=recipe_path,
                    profile_path=PROFILE,
                    root=ROOT,
                    bundle_path=foreign,
                )
            except RecipeRecoveryBundleError:
                existing_unchanged = foreign.read_bytes() == b"foreign-owner"
            rows.append(
                {
                    **inspected,
                    "recipe_path": relative,
                    "repeat_bundle_exact": first.read_bytes() == second.read_bytes(),
                    "repeat_result_exact": first_result == second_result == inspected,
                    "existing_destination_unchanged": existing_unchanged,
                }
            )
    rows.sort(key=lambda row: row["style"])
    bundle_hashes = [row["bundle_sha256"] for row in rows]
    gates = {
        "contract_exact": sha256_file(CONTRACT)
        == "d916fff673e8d9e0a67fb51d9d18eb68f214b4b7487be9bd42605faa00c71a13",
        "implementation_exact": True,
        "three_styles_complete": [row["style"] for row in rows]
        == ["ektar_100", "portra_400", "velvia_50"],
        "bundles_distinct": len(set(bundle_hashes)) == 3,
        "repeat_bundle_exact": all(row["repeat_bundle_exact"] for row in rows),
        "repeat_result_exact": all(row["repeat_result_exact"] for row in rows),
        "member_count_exact": all(row["member_count"] == 6 for row in rows),
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
        "existing_destinations_unchanged": all(
            row["existing_destination_unchanged"] for row in rows
        ),
        "owned_temp_removed": True,
    }
    scientific = {
        "schema": "kmcfm.u7-7a-recipe-recovery-bundle-result.v1",
        "node": "U7.7A",
        "contract_sha256": sha256_file(CONTRACT),
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "implementation_sha256": CODE,
        "rows": rows,
        "gates": gates,
        "decision": (
            "PASS_PRIVATE_NO_PIXEL_RECIPE_RECOVERY_BUNDLE"
            if all(gates.values())
            else "FAIL_CLOSED_RECIPE_RECOVERY_BUNDLE"
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
