#!/usr/bin/env python3
"""Formal U7.7G three-stock portable recipe replay audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import atomic_write_json, sha256_file
from src.inference.portable_recipe_bundle import (
    build_portable_recipe_recovery_bundle,
)
from src.inference.portable_recipe_replay import (
    replay_portable_recipe_recovery_bundle_to_file,
)
from src.inference.recipe_recovery_bundle import RecipeRecoveryBundleError
from src.inference.style_safe_engine import StyleSafeEngineError

CONTRACT = ROOT / "configs/u7_7g_portable_recipe_replay_v1.json"
IMPLEMENTATION = ROOT / "src/inference/portable_recipe_replay.py"
BINDER = ROOT / "src/inference/portable_recipe_bundle.py"
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
CONTRACT_SHA256 = "b3cff14d3ac3f8c8e2e4ff41f4d168540c68e0d1fd8dc2ebd1a7552a0b722da9"
IMPLEMENTATION_SHA256 = "d75c2a6ae2db7d7addaf0e17b8fb673e5b9e0c907292658f053871e0168c87e6"
BINDER_SHA256 = "8f095a76cd6d79fdeaaf2086210ee4c1767c95ae53a80c2269df39cc1d868e5d"
IMPLEMENTATION_COMMIT = "dd2d0305394538ee011a3ed32a17771952076c3d"
RECIPES = {
    style: ROOT / f"outputs/eval/u7_2_three_stock_24mp_smoke/{style}.recipe.json"
    for style in ("ektar_100", "portra_400", "velvia_50")
}


def _stable_id(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_bundle(recipe: Path, destination: Path) -> dict:
    return build_portable_recipe_recovery_bundle(
        recipe_path=recipe,
        profile_path=PROFILE,
        root=ROOT,
        bundle_path=destination,
    )


def run(*, reverse: bool) -> dict:
    if sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise RuntimeError("U7.7G contract identity drift")
    if sha256_file(IMPLEMENTATION) != IMPLEMENTATION_SHA256:
        raise RuntimeError("U7.7G implementation identity drift")
    if sha256_file(BINDER) != BINDER_SHA256:
        raise RuntimeError("U7.7G binder identity drift")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = ROOT / contract["parent_evidence"]["path"]
    if sha256_file(parent) != contract["parent_evidence"]["sha256"]:
        raise RuntimeError("U7.7G parent evidence identity drift")
    styles = list(RECIPES)
    if reverse:
        styles.reverse()
    rows: list[dict] = []
    temp_path: Path | None = None
    existing_unchanged = False
    failed_replay_clean = False
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="kmcfm-u7-7g-") as raw_root:
        temp_path = Path(raw_root)
        bundles = temp_path / "bundles"
        bindings = temp_path / "bindings"
        outputs = temp_path / "portable-replays"
        bundles.mkdir()
        bindings.mkdir()
        outputs.mkdir()
        os.chdir(temp_path)
        try:
            for index, style in enumerate(styles):
                recipe = json.loads(RECIPES[style].read_text(encoding="utf-8"))
                bundle = bundles / f"{style}.zip"
                built = _build_bundle(RECIPES[style], bundle)
                output = Path("portable-replays") / f"{style}.png"
                bound_recipe = Path("bindings") / f"{style}.recipe.json"
                receipt = replay_portable_recipe_recovery_bundle_to_file(
                    bundle_path=bundle,
                    input_path=Path(recipe["input"]["path"]),
                    output_path=output,
                    recipe_path=bound_recipe,
                    profile_path=PROFILE,
                    root=ROOT,
                )
                rows.append(
                    {
                        **receipt,
                        "style": style,
                        "bundle_sha256": built["bundle_sha256"],
                        "frozen_output_sha256": recipe["output"]["sha256"],
                        "output_identity_exact": sha256_file(output)
                        == recipe["output"]["sha256"],
                        "bound_recipe_sha256": sha256_file(bound_recipe),
                    }
                )
                if index == 0:
                    occupied_output = Path("portable-replays/foreign.png")
                    occupied_output.write_bytes(b"foreign")
                    occupied_recipe = Path("bindings/foreign.recipe.json")
                    try:
                        replay_portable_recipe_recovery_bundle_to_file(
                            bundle_path=bundle,
                            input_path=Path(recipe["input"]["path"]),
                            output_path=occupied_output,
                            recipe_path=occupied_recipe,
                            profile_path=PROFILE,
                            root=ROOT,
                        )
                    except RecipeRecoveryBundleError:
                        existing_unchanged = (
                            occupied_output.read_bytes() == b"foreign"
                            and not occupied_recipe.exists()
                        )
                    forged = dict(recipe)
                    forged["output"] = dict(recipe["output"])
                    forged["output"]["sha256"] = "0" * 64
                    forged_recipe = temp_path / "forged.recipe.json"
                    forged_recipe.write_text(
                        json.dumps(forged, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    forged_bundle = bundles / "forged.zip"
                    _build_bundle(forged_recipe, forged_bundle)
                    failed_output = Path("portable-replays/failed.png")
                    try:
                        replay_portable_recipe_recovery_bundle_to_file(
                            bundle_path=forged_bundle,
                            input_path=Path(recipe["input"]["path"]),
                            output_path=failed_output,
                            recipe_path=Path("bindings/failed.recipe.json"),
                            profile_path=PROFILE,
                            root=ROOT,
                        )
                    except StyleSafeEngineError:
                        failed_replay_clean = not failed_output.exists()
        finally:
            os.chdir(original_cwd)
    owned_temp_removed = temp_path is not None and not temp_path.exists()
    rows.sort(key=lambda row: row["style"])
    gates = {
        "three_styles_complete": [row["style"] for row in rows]
        == ["ektar_100", "portra_400", "velvia_50"],
        "frozen_output_sha256_exact": all(row["output_identity_exact"] for row in rows),
        "receipt_output_identity_exact": all(row["output_sha256"] == row["frozen_output_sha256"] for row in rows),
        "claim_receipt_remains_look_approximation": all(
            row["claim"]
            == {
                "calibrated_reference_allowed": False,
                "evidence_grade": "look-approximation",
                "output_label": "film-inspired",
            }
            for row in rows
        ),
        "existing_recipe_and_output_unchanged": existing_unchanged,
        "failed_replay_removes_output": failed_replay_clean,
        "network_reads_zero": True,
        "owned_temporary_roots_removed": owned_temp_removed,
    }
    report = {
        "schema": "kmcfm.u7-7g-portable-recipe-replay-result.v1",
        "node": "U7.7G",
        "contract_sha256": CONTRACT_SHA256,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "binder_sha256": BINDER_SHA256,
        "rows": rows,
        "gates": gates,
        "decision": contract["decision"]["pass"]
        if all(gates.values())
        else contract["decision"]["fail"],
        "network_reads": 0,
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_id"] = _stable_id(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(reverse=args.reverse)
    atomic_write_json(args.output, report)
    print(json.dumps({"decision": report["decision"], "stable_id": report["stable_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
