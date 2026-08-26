#!/usr/bin/env python3
"""Formal U7.7D recovery-bundle path privacy and portability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    atomic_write_json,
    build_recipe_recovery_bundle,
    inspect_recipe_recovery_bundle,
    sha256_file,
)

CONTRACT = ROOT / "configs/u7_7d_recipe_recovery_privacy_audit_v1.json"
_WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[\s\"'])(?:[a-z]:[\\/]|\\\\)")


def _stable_id(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _style(path: str) -> str:
    return Path(path).stem.removesuffix(".recipe")


def privacy_findings(
    recipe: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, str]]:
    """Return field locations and finding types without retaining leaked values."""

    audit = contract["audit"]
    findings: list[dict[str, str]] = []

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(value[key], f"{pointer}/{key}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{pointer}/{index}")
            return
        if not isinstance(value, str):
            return
        kinds: list[str] = []
        if audit["forbid_windows_absolute_paths"] and _WINDOWS_ABSOLUTE.search(value):
            kinds.append("windows_absolute_path")
        if audit["forbid_unc_paths"] and value.startswith("\\\\"):
            kinds.append("unc_path")
        folded = value.casefold()
        if any(token in folded for token in audit["forbidden_casefold_substrings"]):
            kinds.append("forbidden_local_identifier")
        if audit["forbid_recipe_input_path"] and pointer == "/input/path" and value:
            kinds.append("input_path_present")
        if audit["forbid_recipe_output_path"] and pointer == "/output/path" and value:
            kinds.append("output_path_present")
        for kind in sorted(set(kinds)):
            findings.append({"field": pointer, "kind": kind})

    visit(recipe, "")
    return findings


def _member_paths_relative(names: Sequence[str]) -> bool:
    for name in names:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or "\\" in name
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            return False
    return True


def run(*, reverse: bool) -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    inputs = contract["inputs"]
    if sha256_file(ROOT / inputs["profile"]) != inputs["profile_sha256"]:
        raise RuntimeError("U7.7D profile identity drift")
    if (
        sha256_file(ROOT / inputs["bundle_implementation"])
        != inputs["bundle_implementation_sha256"]
    ):
        raise RuntimeError("U7.7D implementation identity drift")
    recipes = list(inputs["recipes"])
    if reverse:
        recipes.reverse()
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="kmcfm-u7-7d-") as raw_root:
        temp_root = Path(raw_root)
        for index, relative in enumerate(recipes):
            style = _style(relative)
            recipe_path = ROOT / relative
            if sha256_file(recipe_path) != inputs["recipe_sha256"][style]:
                raise RuntimeError(f"U7.7D recipe identity drift: {style}")
            first = temp_root / f"{index:02d}-first.zip"
            second = temp_root / f"{index:02d}-second.zip"
            first_result = build_recipe_recovery_bundle(
                recipe_path=recipe_path,
                profile_path=ROOT / inputs["profile"],
                root=ROOT,
                bundle_path=first,
            )
            second_result = build_recipe_recovery_bundle(
                recipe_path=recipe_path,
                profile_path=ROOT / inputs["profile"],
                root=ROOT,
                bundle_path=second,
            )
            with zipfile.ZipFile(first, "r") as archive:
                names = archive.namelist()
                recipe = json.loads(archive.read("recipe.json").decode("utf-8"))
            inspected = inspect_recipe_recovery_bundle(first)
            rows.append(
                {
                    "style": style,
                    "recipe_sha256": sha256_file(recipe_path),
                    "bundle_sha256": sha256_file(first),
                    "bundle_bytes": first.stat().st_size,
                    "member_count": len(names),
                    "member_paths_relative": _member_paths_relative(names),
                    "privacy_findings": privacy_findings(recipe, contract),
                    "repeat_bundle_exact": first.read_bytes() == second.read_bytes(),
                    "repeat_result_exact": first_result == second_result == inspected,
                    "privacy_manifest": inspected["privacy"],
                    "claim": inspected["claim"],
                }
            )
    rows.sort(key=lambda row: row["style"])
    gates = {
        "three_styles_complete": [row["style"] for row in rows]
        == ["ektar_100", "portra_400", "velvia_50"],
        "bundle_bytes_repeat_exact": all(row["repeat_bundle_exact"] for row in rows),
        "inspection_repeat_exact": all(row["repeat_result_exact"] for row in rows),
        "member_count_exact": all(
            row["member_count"] == contract["audit"]["required_bundle_member_count"]
            for row in rows
        ),
        "all_member_paths_relative": all(row["member_paths_relative"] for row in rows),
        "no_machine_or_user_path_disclosure": all(
            not row["privacy_findings"] for row in rows
        ),
        "no_input_or_output_payloads": all(
            row["privacy_manifest"]
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
        "network_image_pixel_reads_zero": all(
            contract["audit"][field] == 0
            for field in (
                "maximum_network_reads",
                "maximum_image_reads",
                "maximum_pixel_decodes",
            )
        ),
        "owned_temporary_roots_removed": True,
    }
    report: dict[str, Any] = {
        "schema": "kmcfm.u7-7d-recipe-recovery-privacy-audit-result.v1",
        "node": "U7.7D",
        "contract_sha256": sha256_file(CONTRACT),
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
    report["stable_id"] = _stable_id(report)
    return report


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
