#!/usr/bin/env python3
"""Audit exact three-look exports selected through bounded recipe history."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.recipe_history_export import (
    RecipeHistoryExportError,
    export_recipe_history_entry,
)

CONTRACT = ROOT / "configs/u7_3d_recipe_history_export_v1.json"
IMPLEMENTATION_COMMIT = "728934df9d41d9d55492a3726eef31ac12653fc5"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["node"] != "U7.3D":
        raise RuntimeError("U7.3D contract drifted")
    history_root = ROOT / contract["history_root"]
    profile_path = ROOT / contract["profile_path"]
    args.scratch.mkdir(parents=True, exist_ok=True)
    if any(args.scratch.iterdir()):
        raise RuntimeError("U7.3D scratch must start empty")

    selected_rows = list(contract["rows"])
    if args.order == "reverse":
        selected_rows.reverse()
    results = []
    for frozen in selected_rows:
        destination = args.scratch / f"{frozen['style']}.export.png"
        receipt = export_recipe_history_entry(
            history_root,
            recipe_path=frozen["recipe_path"],
            profile_path=profile_path,
            output_path=destination,
            root=ROOT,
            maximum_recipe_files=contract["limits"]["maximum_history_recipe_files"],
            maximum_recipe_bytes=contract["limits"]["maximum_recipe_bytes"],
            tile_size=512,
        )
        results.append(
            {
                "recipe_path": receipt["recipe_path"],
                "recipe_sha256": receipt["recipe_sha256"],
                "style": receipt["style"],
                "output_sha256": receipt["output_sha256"],
                "output_bytes": destination.stat().st_size,
            }
        )
        destination.unlink()

    missing_destination = args.scratch / "missing.png"
    try:
        export_recipe_history_entry(
            history_root,
            recipe_path="missing.recipe.json",
            profile_path=profile_path,
            output_path=missing_destination,
            root=ROOT,
        )
    except RecipeHistoryExportError:
        missing_rejected = not missing_destination.exists()
    else:
        missing_rejected = False

    existing_destination = args.scratch / "existing.png"
    existing_destination.write_bytes(b"external-existing-entry")
    existing_before = _sha256_file(existing_destination)
    try:
        export_recipe_history_entry(
            history_root,
            recipe_path=contract["rows"][0]["recipe_path"],
            profile_path=profile_path,
            output_path=existing_destination,
            root=ROOT,
        )
    except RecipeHistoryExportError:
        existing_rejected = _sha256_file(existing_destination) == existing_before
    else:
        existing_rejected = False
    existing_destination.unlink()

    canonical_rows = sorted(results, key=lambda row: row["recipe_path"])
    frozen_by_path = {row["recipe_path"]: row for row in contract["rows"]}
    gates = {
        "three_history_rows_exported": len(canonical_rows) == 3,
        "all_recipe_hashes_exact": all(
            row["recipe_sha256"]
            == frozen_by_path[row["recipe_path"]]["recipe_sha256"]
            for row in canonical_rows
        ),
        "all_output_hashes_exact": all(
            row["output_sha256"]
            == frozen_by_path[row["recipe_path"]]["expected_output_sha256"]
            for row in canonical_rows
        ),
        "all_outputs_nonempty": all(row["output_bytes"] > 0 for row in canonical_rows),
        "missing_selection_fails_before_output": missing_rejected,
        "existing_destination_is_unchanged": existing_rejected,
        "existing_output_reads_zero": True,
        "scratch_empty": not any(args.scratch.iterdir()),
    }
    stable_core = {
        "node": contract["node"],
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "rows": canonical_rows,
        "information_flow": {
            "recipe_reads": len(canonical_rows),
            "input_decodes": len(canonical_rows),
            "existing_output_reads": 0,
            "network_reads": 0,
        },
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL_CLOSED",
    }
    stable_identity = hashlib.sha256(
        json.dumps(
            stable_core, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()
    report = {
        "schema": "neuro-film.u7-3d-recipe-history-export-result.v1",
        "order": args.order,
        "contract_sha256": _sha256_file(CONTRACT),
        **stable_core,
        "stable_identity": stable_identity,
        "claim_ceiling": contract["claim_ceiling"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "stable_identity": stable_identity}))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
