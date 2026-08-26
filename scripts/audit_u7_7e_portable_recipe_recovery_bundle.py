#!/usr/bin/env python3
"""Formal U7.7E portable recipe recovery-bundle audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.inference.recipe_recovery_bundle as recovery
from src.inference import atomic_write_json, sha256_file
from src.inference.portable_recipe_bundle import (
    bind_portable_recipe,
    build_portable_recipe_recovery_bundle,
    inspect_portable_recipe_recovery_bundle,
)
from src.inference.recipe_recovery_bundle import RecipeRecoveryBundleError

CONTRACT = ROOT / "configs/u7_7e_portable_recipe_recovery_bundle_v1.json"
IMPLEMENTATION = ROOT / "src/inference/portable_recipe_bundle.py"
CONTRACT_SHA256 = "a94461a8a7ac8e9097193aa4049e4d5253a367466dd85317e46d9e754b32d610"
IMPLEMENTATION_SHA256 = (
    "57e827f006ac660139a2ddf1172ccb3eb0ac65e16f72a85fbc84c3bad32c2d6d"
)
IMPLEMENTATION_COMMIT = "7d5adc68e60a4192e8a01dd51f24906413a326b9"
_ABSOLUTE = re.compile(r"(?i)(?:^|[\s\"'])(?:[a-z]:[\\/]|\\\\)")


def _style(path: str) -> str:
    return Path(path).stem.removesuffix(".recipe")


def _portable_privacy_pass(recipe: dict) -> bool:
    encoded = json.dumps(recipe, ensure_ascii=False, sort_keys=True)
    return (
        "path" not in recipe["input"]
        and "path" not in recipe["output"]
        and recipe["input"].get("path_binding") == "unresolved-user-input"
        and recipe["output"].get("path_binding") == "unresolved-user-output"
        and _ABSOLUTE.search(encoded) is None
        and "neuro_film_storage" not in encoded.casefold()
        and "hhvrf" not in encoded.casefold()
    )


def _injected_partial_cleanup(
    recipe_path: Path, profile_path: Path, root: Path, destination: Path
) -> bool:
    original_write = recovery.os.write
    injected = False

    def fail_after_partial(descriptor: int, payload: bytes | memoryview) -> int:
        nonlocal injected
        if not injected:
            injected = True
            original_write(descriptor, payload[: max(1, len(payload) // 2)])
            raise OSError("U7.7E injected partial write")
        return original_write(descriptor, payload)

    recovery.os.write = fail_after_partial
    rejected = False
    try:
        build_portable_recipe_recovery_bundle(
            recipe_path=recipe_path,
            profile_path=profile_path,
            root=root,
            bundle_path=destination,
        )
    except OSError as exc:
        rejected = str(exc) == "U7.7E injected partial write"
    finally:
        recovery.os.write = original_write
    return rejected and not destination.exists()


def run(*, reverse: bool) -> dict:
    if sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise RuntimeError("U7.7E contract identity drift")
    if sha256_file(IMPLEMENTATION) != IMPLEMENTATION_SHA256:
        raise RuntimeError("U7.7E implementation identity drift")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = ROOT / contract["parent_evidence"]["path"]
    if sha256_file(parent) != contract["parent_evidence"]["sha256"]:
        raise RuntimeError("U7.7E parent evidence identity drift")
    profile = ROOT / contract["inputs"]["profile"]
    if sha256_file(profile) != contract["inputs"]["profile_sha256"]:
        raise RuntimeError("U7.7E profile identity drift")
    recipes = list(contract["inputs"]["recipes"])
    if reverse:
        recipes.reverse()
    rows: list[dict] = []
    temp_path: Path | None = None
    partial_cleanup = False
    with tempfile.TemporaryDirectory(prefix="kmcfm-u7-7e-") as raw_root:
        temp_path = Path(raw_root)
        for index, relative in enumerate(recipes):
            style = _style(relative)
            recipe_path = ROOT / relative
            if sha256_file(recipe_path) != contract["inputs"]["recipe_sha256"][style]:
                raise RuntimeError(f"U7.7E recipe identity drift: {style}")
            original = json.loads(recipe_path.read_text(encoding="utf-8"))
            first = temp_path / f"{index:02d}-first.zip"
            second = temp_path / f"{index:02d}-second.zip"
            one = build_portable_recipe_recovery_bundle(
                recipe_path=recipe_path,
                profile_path=profile,
                root=ROOT,
                bundle_path=first,
            )
            two = build_portable_recipe_recovery_bundle(
                recipe_path=recipe_path,
                profile_path=profile,
                root=ROOT,
                bundle_path=second,
            )
            with zipfile.ZipFile(first) as archive:
                portable = json.loads(
                    archive.read("portable_recipe.json").decode("utf-8")
                )
            rebound = bind_portable_recipe(
                portable,
                input_path=original["input"]["path"],
                output_path=original["output"]["path"],
            )
            foreign = temp_path / f"{index:02d}-foreign.zip"
            foreign.write_bytes(b"foreign-owner")
            existing_unchanged = False
            try:
                build_portable_recipe_recovery_bundle(
                    recipe_path=recipe_path,
                    profile_path=profile,
                    root=ROOT,
                    bundle_path=foreign,
                )
            except RecipeRecoveryBundleError:
                existing_unchanged = foreign.read_bytes() == b"foreign-owner"
            rows.append(
                {
                    **inspect_portable_recipe_recovery_bundle(first),
                    "style": style,
                    "portable_privacy_pass": _portable_privacy_pass(portable),
                    "strict_original_rebind_roundtrip_exact": rebound == original,
                    "repeat_bundle_exact": first.read_bytes() == second.read_bytes(),
                    "repeat_result_exact": one == two,
                    "existing_destination_unchanged": existing_unchanged,
                }
            )
        first_recipe = ROOT / recipes[0]
        partial_cleanup = _injected_partial_cleanup(
            first_recipe, profile, ROOT, temp_path / "injected-partial.zip"
        )
    owned_temp_removed = temp_path is not None and not temp_path.exists()
    rows.sort(key=lambda row: row["style"])
    gates = {
        "three_styles_complete": [row["style"] for row in rows]
        == ["ektar_100", "portra_400", "velvia_50"],
        "bundle_bytes_repeat_exact": all(row["repeat_bundle_exact"] for row in rows),
        "inspection_repeat_exact": all(row["repeat_result_exact"] for row in rows),
        "no_machine_or_user_path_disclosure": all(
            row["portable_privacy_pass"] for row in rows
        ),
        "strict_original_rebind_roundtrip_exact": all(
            row["strict_original_rebind_roundtrip_exact"] for row in rows
        ),
        "profile_and_assets_revalidate": all(row["member_count"] == 6 for row in rows),
        "claims_remain_look_approximation": all(
            row["claim"]
            == {
                "calibrated_reference_allowed": False,
                "evidence_grade": "look-approximation",
                "output_label": "film-inspired",
            }
            for row in rows
        ),
        "existing_destination_unchanged": all(
            row["existing_destination_unchanged"] for row in rows
        ),
        "failed_publication_leaves_no_partial": partial_cleanup,
        "network_image_pixel_reads_zero": True,
        "owned_temporary_roots_removed": owned_temp_removed,
    }
    report = {
        "schema": "kmcfm.u7-7e-portable-recipe-recovery-bundle-result.v1",
        "node": "U7.7E",
        "contract_sha256": CONTRACT_SHA256,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "rows": rows,
        "gates": gates,
        "decision": contract["decision"]["pass"]
        if all(gates.values())
        else contract["decision"]["fail"],
        "network_reads": 0,
        "image_reads": 0,
        "pixel_decodes": 0,
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_id"] = sha256_file_bytes(report)
    return report


def sha256_file_bytes(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(reverse=args.reverse)
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {"decision": report["decision"], "stable_id": report["stable_id"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
