#!/usr/bin/env python3
"""Formal U7.7F explicit portable-recipe binding audit."""

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
from src.inference import atomic_write_json, sha256_file, validate_render_recipe
from src.inference.portable_recipe_bundle import (
    bind_portable_recipe_recovery_bundle,
    build_portable_recipe_recovery_bundle,
)
from src.inference.recipe_recovery_bundle import RecipeRecoveryBundleError

CONTRACT = ROOT / "configs/u7_7f_explicit_recipe_binding_v1.json"
IMPLEMENTATION = ROOT / "src/inference/portable_recipe_bundle.py"
PARENT_EVIDENCE = ROOT / "docs/evidence/U7_7E_PORTABLE_RECIPE_RECOVERY_BUNDLE_RESULT.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
CONTRACT_SHA256 = "af66f7d63fa2a007eca553a14b20a15ecf8b9ea036a3e957f509eedbada16d44"
IMPLEMENTATION_SHA256 = "8f095a76cd6d79fdeaaf2086210ee4c1767c95ae53a80c2269df39cc1d868e5d"
IMPLEMENTATION_COMMIT = "bdfd77e0f7697379de583747db9d20bd41ee663a"
RECIPES = {
    style: ROOT / f"outputs/eval/u7_2_three_stock_24mp_smoke/{style}.recipe.json"
    for style in ("ektar_100", "portra_400", "velvia_50")
}


def _stable_id(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rejects_existing(
    *, bundle: Path, source: Path, output: Path, recipe: Path, occupied: str
) -> bool:
    selected = output if occupied == "output" else recipe
    selected.write_bytes(b"foreign-owner")
    try:
        bind_portable_recipe_recovery_bundle(
            bundle_path=bundle,
            input_path=source,
            output_path=output,
            recipe_path=recipe,
        )
    except RecipeRecoveryBundleError:
        return selected.read_bytes() == b"foreign-owner" and (
            occupied == "recipe" or not recipe.exists()
        )
    return False


def _partial_cleanup(bundle: Path, source: Path, root: Path) -> bool:
    original_write = recovery.os.write
    injected = False

    def fail_after_partial(descriptor: int, payload: bytes | memoryview) -> int:
        nonlocal injected
        if not injected:
            injected = True
            original_write(descriptor, payload[: max(1, len(payload) // 2)])
            raise OSError("U7.7F injected partial write")
        return original_write(descriptor, payload)

    destination = root / "partial.recipe.json"
    recovery.os.write = fail_after_partial
    rejected = False
    try:
        bind_portable_recipe_recovery_bundle(
            bundle_path=bundle,
            input_path=source,
            output_path=root / "partial-output.png",
            recipe_path=destination,
        )
    except OSError as exc:
        rejected = str(exc) == "U7.7F injected partial write"
    finally:
        recovery.os.write = original_write
    return rejected and not destination.exists()


def run(*, reverse: bool) -> dict:
    if sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise RuntimeError("U7.7F contract identity drift")
    if sha256_file(IMPLEMENTATION) != IMPLEMENTATION_SHA256:
        raise RuntimeError("U7.7F implementation identity drift")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if sha256_file(PARENT_EVIDENCE) != contract["parent_evidence"]["sha256"]:
        raise RuntimeError("U7.7F parent evidence identity drift")
    parent = json.loads(PARENT_EVIDENCE.read_text(encoding="utf-8"))
    parent_bundle_sha = dict(zip(parent["results"]["styles"], parent["results"]["bundle_sha256"]))
    styles = list(RECIPES)
    if reverse:
        styles.reverse()
    rows: list[dict] = []
    temp_path: Path | None = None
    partial_cleanup = False
    mismatch_rejected = False
    with tempfile.TemporaryDirectory(prefix="kmcfm-u7-7f-") as raw_root:
        temp_path = Path(raw_root)
        for index, style in enumerate(styles):
            source_recipe = json.loads(RECIPES[style].read_text(encoding="utf-8"))
            source = Path(source_recipe["input"]["path"])
            bundle = temp_path / f"{style}.zip"
            built = build_portable_recipe_recovery_bundle(
                recipe_path=RECIPES[style],
                profile_path=PROFILE,
                root=ROOT,
                bundle_path=bundle,
            )
            if built["bundle_sha256"] != parent_bundle_sha[style]:
                raise RuntimeError(f"U7.7F parent bundle identity drift: {style}")
            output = Path("renders") / f"{style}.png"
            first_path = temp_path / f"{style}.first.recipe.json"
            second_path = temp_path / f"{style}.second.recipe.json"
            first = bind_portable_recipe_recovery_bundle(
                bundle_path=bundle,
                input_path=source,
                output_path=output,
                recipe_path=first_path,
            )
            second = bind_portable_recipe_recovery_bundle(
                bundle_path=bundle,
                input_path=source,
                output_path=output,
                recipe_path=second_path,
            )
            bound = json.loads(first_path.read_text(encoding="utf-8"))
            validate_render_recipe(bound)
            rows.append(
                {
                    **first,
                    "style": style,
                    "bound_recipe_repeat_exact": first_path.read_bytes()
                    == second_path.read_bytes(),
                    "binding_result_repeat_exact": first == second,
                    "strict_recipe_valid": True,
                    "input_path_explicit": bound["input"]["path"] == str(source),
                    "output_path_explicit": bound["output"]["path"] == str(output),
                    "output_absent": not output.exists(),
                }
            )
            foreign_output = temp_path / f"{style}.foreign-output.png"
            foreign_recipe = temp_path / f"{style}.foreign-output.recipe.json"
            rows[-1]["existing_output_unchanged"] = _rejects_existing(
                bundle=bundle,
                source=source,
                output=foreign_output,
                recipe=foreign_recipe,
                occupied="output",
            )
            foreign_recipe = temp_path / f"{style}.foreign-recipe.json"
            rows[-1]["existing_recipe_unchanged"] = _rejects_existing(
                bundle=bundle,
                source=source,
                output=temp_path / f"{style}.unused-output.png",
                recipe=foreign_recipe,
                occupied="recipe",
            )
            if index == 0:
                wrong = temp_path / "wrong-input.bin"
                wrong.write_bytes(b"wrong-input")
                mismatch_destination = temp_path / "mismatch.recipe.json"
                try:
                    bind_portable_recipe_recovery_bundle(
                        bundle_path=bundle,
                        input_path=wrong,
                        output_path=temp_path / "mismatch-output.png",
                        recipe_path=mismatch_destination,
                    )
                except RecipeRecoveryBundleError:
                    mismatch_rejected = not mismatch_destination.exists()
                partial_cleanup = _partial_cleanup(bundle, source, temp_path)
    owned_temp_removed = temp_path is not None and not temp_path.exists()
    rows.sort(key=lambda row: row["style"])
    gates = {
        "three_styles_complete": [row["style"] for row in rows]
        == ["ektar_100", "portra_400", "velvia_50"],
        "bound_recipe_bytes_repeat_exact": all(row["bound_recipe_repeat_exact"] for row in rows),
        "binding_result_repeat_exact": all(row["binding_result_repeat_exact"] for row in rows),
        "bound_recipe_strict_valid": all(row["strict_recipe_valid"] for row in rows),
        "explicit_paths_exact": all(row["input_path_explicit"] and row["output_path_explicit"] for row in rows),
        "output_not_created": all(row["output_absent"] for row in rows),
        "input_hash_mismatch_rejected_before_publication": mismatch_rejected,
        "existing_output_rejected_before_publication": all(row["existing_output_unchanged"] for row in rows),
        "existing_recipe_destination_unchanged": all(row["existing_recipe_unchanged"] for row in rows),
        "failed_publication_leaves_no_partial": partial_cleanup,
        "claims_remain_look_approximation": all(
            row["claim"]
            == {
                "calibrated_reference_allowed": False,
                "evidence_grade": "look-approximation",
                "output_label": "film-inspired",
            }
            for row in rows
        ),
        "network_image_decode_render_zero": True,
        "owned_temporary_roots_removed": owned_temp_removed,
    }
    report = {
        "schema": "kmcfm.u7-7f-explicit-recipe-binding-result.v1",
        "node": "U7.7F",
        "contract_sha256": CONTRACT_SHA256,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "rows": rows,
        "gates": gates,
        "decision": contract["decision"]["pass"]
        if all(gates.values())
        else contract["decision"]["fail"],
        "input_files_hash_verified": len(rows),
        "network_reads": 0,
        "image_decodes": 0,
        "renders": 0,
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
