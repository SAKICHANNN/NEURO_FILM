#!/usr/bin/env python3
"""Audit product recipe commit reuse against a transient live-HEAD probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import render_film

CONFIG = ROOT / "configs/u7_21d_product_recipe_commit_snapshot_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> None:
    yy, xx = np.mgrid[:31, :47]
    rgb = np.stack(
        (
            (xx * 5 + yy * 3) % 256,
            (xx * 7 + yy * 11) % 256,
            (xx * 13 + yy * 2) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _case(root: Path, case: str, config: dict[str, object]) -> dict[str, object]:
    case_root = root / case
    case_root.mkdir()
    source = case_root / "source.png"
    output = case_root / "output.png"
    _source(source)
    frozen = str(config["frozen_commit"])
    transient = str(config["transient_commit"])
    selection: tuple[str, ...]
    if case == "shortcut":
        selection = ("--product-look", "ektar_100")
    elif case == "explicit-profile":
        selection = (
            "--use-render-profile",
            "--render-profile",
            str(ROOT / "configs/render_profiles/safe_rich_product_v1.json"),
            "--style",
            "ektar_100",
        )
    else:
        selection = ()
    live_calls = 0

    def live_head(*_args: object, **_kwargs: object) -> str:
        nonlocal live_calls
        live_calls += 1
        return transient + "\n"

    argv = [
        "render_film.py",
        str(source),
        *selection,
        "--output",
        str(output),
        "--write-recipe",
    ]
    with (
        mock.patch.object(render_film, "_source_commit", return_value=frozen),
        mock.patch.object(render_film, "_validate_runtime_source_scope"),
        mock.patch.object(render_film.subprocess, "check_output", side_effect=live_head),
        mock.patch.object(sys, "argv", argv),
    ):
        returncode = render_film.main()
    recipe_path = output.with_suffix(".recipe.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    return {
        "case": case,
        "returncode": returncode,
        "recipe_commit": recipe["software"]["commit"],
        "live_head_calls": live_calls,
        "output_sha256": _sha256(output),
        "recipe_claim": recipe["claim"],
        "stage_residue": sorted(path.name for path in case_root.glob(".*.stage*")),
    }


def audit(scratch_root: Path, order: str) -> dict[str, object]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    requested = ["shortcut", "explicit-profile", "legacy"]
    if order == "reverse":
        requested.reverse()
    with tempfile.TemporaryDirectory(prefix="u7-21d-", dir=scratch_root) as temporary:
        rows = [_case(Path(temporary), case, config) for case in requested]
    cases = sorted(rows, key=lambda row: str(row["case"]))
    by_case = {str(row["case"]): row for row in cases}
    frozen = config["frozen_commit"]
    transient = config["transient_commit"]
    gates = {
        "product_recipes_use_frozen_commit": all(
            by_case[name]["recipe_commit"] == frozen
            for name in ("shortcut", "explicit-profile")
        ),
        "product_recipe_live_head_reads_zero": all(
            by_case[name]["live_head_calls"] == 0
            for name in ("shortcut", "explicit-profile")
        ),
        "legacy_recipe_uses_live_commit": by_case["legacy"]["recipe_commit"]
        == transient,
        "legacy_live_head_reads_one": by_case["legacy"]["live_head_calls"] == 1,
        "all_commands_succeed": all(row["returncode"] == 0 for row in cases),
        "stage_residue_zero": all(not row["stage_residue"] for row in cases),
        "claim_ceiling_exact": all(
            row["recipe_claim"]["output_label"] == "film-inspired"
            and row["recipe_claim"]["evidence_grade"] == "look-approximation"
            and row["recipe_claim"]["calibrated_reference_allowed"] is False
            for row in cases
        ),
    }
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    return {
        "schema": "kmcfm.u7-21d-product-recipe-commit-snapshot-result.v1",
        "status": (
            "PASS_PRIVATE_U7_21D_PRODUCT_RECIPE_COMMIT_SNAPSHOT"
            if all(gates.values())
            else "FAIL_CLOSED_U7_21D_PRODUCT_RECIPE_COMMIT_SNAPSHOT"
        ),
        "source_commit": source_commit,
        "cases": cases,
        "gates": gates,
        "claim_ceiling": config["claim"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.scratch_root.resolve(strict=True), args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if str(report["status"]).startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
