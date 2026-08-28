#!/usr/bin/env python3
"""Audit authoritative product-catalog enforcement for safe-rich-product-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "tmp/u7_2h_prechange"
SCRIPT = ROOT / "scripts/render_film.py"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
LEGACY_PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
U7_2H_CONFIG = ROOT / "configs/u7_2h_explicit_product_look_selection_v1.json"
U7_2H_EVIDENCE = ROOT / "docs/evidence/U7_2H_EXPLICIT_PRODUCT_LOOK_SELECTION_RESULT.json"
CATALOG = ROOT / "src/inference/product_look_catalog.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: Any) -> str:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalized_recipe(path: Path, *, portable_paths: bool) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["software"]["commit"] = "<normalized>"
    if portable_paths:
        payload["input"]["path"] = "<input>"
        payload["output"]["path"] = "<output>"
    return _canonical_sha(payload)


def _source(path: Path) -> None:
    y, x = np.mgrid[:47, :61]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _run(source: Path, output: Path, arguments: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(source), *arguments, "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _message(stderr: str) -> str:
    needles = (
        "only supports available product-catalog looks",
        "product look 'generic_bw' is unavailable",
    )
    return next(
        (line.strip() for line in stderr.splitlines() if any(item in line for item in needles)),
        "",
    )


def build_report(*, config_path: Path, order: tuple[str, ...]) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_config = json.loads(U7_2H_CONFIG.read_text(encoding="utf-8"))
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    WORK_ROOT.mkdir(parents=True)
    source = WORK_ROOT / "source.png"
    _source(source)
    source_before = _sha(source)
    product_results: dict[str, dict[str, Any]] = {}
    rejection_results: dict[str, dict[str, Any]] = {}
    try:
        for style in order:
            output = WORK_ROOT / f"{style}.png"
            completed = _run(
                source,
                output,
                (
                    "--style",
                    style,
                    "--use-render-profile",
                    "--render-profile",
                    str(PRODUCT_PROFILE),
                    "--write-recipe",
                ),
            )
            recipe = output.with_suffix(".recipe.json")
            product_results[style] = {
                "returncode": completed.returncode,
                "output_sha256": _sha(output) if output.exists() else None,
                "normalized_recipe_sha256": (
                    _normalized_recipe(recipe, portable_paths=False)
                    if recipe.exists()
                    else None
                ),
            }

        invalid_order = list(config["non_product_profile_styles"]) + ["unknown", ""]
        if order != tuple(config["available_product_looks"]):
            invalid_order.reverse()
        for index, style in enumerate(invalid_order):
            output = WORK_ROOT / f"rejected_{index}.png"
            completed = _run(
                WORK_ROOT / "must_not_decode.png",
                output,
                (
                    "--style",
                    style,
                    "--use-render-profile",
                    "--render-profile",
                    str(PRODUCT_PROFILE),
                    "--write-recipe",
                ),
            )
            rejection_results[style or "<empty>"] = {
                "returncode": completed.returncode,
                "message": _message(completed.stderr),
                "input_decode_attempted": "must_not_decode" in completed.stderr,
                "residue_count": sum(
                    path.exists()
                    for path in (output, output.with_suffix(".recipe.json"))
                ),
            }

        blocked_output = WORK_ROOT / "blocked_generic_bw.png"
        blocked = _run(
            WORK_ROOT / "must_not_decode.png",
            blocked_output,
            (
                "--style",
                "generic_bw",
                "--use-render-profile",
                "--render-profile",
                str(PRODUCT_PROFILE),
                "--write-recipe",
            ),
        )
        rejection_results["generic_bw"] = {
            "returncode": blocked.returncode,
            "message": _message(blocked.stderr),
            "input_decode_attempted": "must_not_decode" in blocked.stderr,
            "residue_count": sum(
                path.exists()
                for path in (
                    blocked_output,
                    blocked_output.with_suffix(".recipe.json"),
                )
            ),
        }

        legacy_output = WORK_ROOT / "legacy_hp5.png"
        legacy = _run(
            source,
            legacy_output,
            (
                "--style",
                "hp5",
                "--use-render-profile",
                "--render-profile",
                str(LEGACY_PROFILE),
                "--write-recipe",
            ),
        )
        legacy_recipe = legacy_output.with_suffix(".recipe.json")
        oracle = parent_config["prechange_oracle"]
        legacy_oracle = config["prechange_legacy_hp5_oracle"]
        product_exact = all(
            product_results[style]["returncode"] == 0
            and product_results[style]["output_sha256"]
            == oracle["product"][style]["output_sha256"]
            and product_results[style]["normalized_recipe_sha256"]
            == oracle["product"][style]["normalized_recipe_sha256"]
            for style in config["available_product_looks"]
        )
        rejection_exact = all(
            row["returncode"] != 0
            and bool(row["message"])
            and not row["input_decode_attempted"]
            and row["residue_count"] == 0
            for row in rejection_results.values()
        )
        gates = {
            "fixture_exact": source_before == legacy_oracle["fixture_sha256"],
            "source_immutable": _sha(source) == source_before,
            "parent_u7_2h_evidence_exact": _sha(U7_2H_EVIDENCE)
            == config["source_locks"]["parent_u7_2h_evidence_sha256"],
            "product_profile_exact": _sha(PRODUCT_PROFILE)
            == config["source_locks"]["product_profile_sha256"],
            "product_catalog_exact": _sha(CATALOG)
            == config["source_locks"]["product_look_catalog_sha256"],
            "cli_changed_from_prechange_lock": _sha(SCRIPT)
            != config["source_locks"]["render_film_sha256"],
            "all_non_catalog_and_blocked_styles_reject_before_decode": rejection_exact,
            "three_available_product_looks_exact": product_exact,
            "legacy_hp5_output_exact": legacy.returncode == 0
            and _sha(legacy_output) == legacy_oracle["output_sha256"],
            "legacy_hp5_recipe_semantics_exact": legacy.returncode == 0
            and _normalized_recipe(legacy_recipe, portable_paths=True)
            == legacy_oracle["normalized_recipe_sha256"],
        }
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
        report = {
            "schema_id": "neuro-film.u7-2i-product-look-catalog-enforcement-result.v1",
            "experiment_id": "U7.2I",
            "implementation_commit": head,
            "bindings": {
                "config": {
                    "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": _sha(config_path),
                },
                "cli": {"path": "scripts/render_film.py", "sha256": _sha(SCRIPT)},
                "catalog": {
                    "path": "src/inference/product_look_catalog.py",
                    "sha256": _sha(CATALOG),
                },
                "product_profile_sha256": _sha(PRODUCT_PROFILE),
                "parent_u7_2h_evidence_sha256": _sha(U7_2H_EVIDENCE),
            },
            "product_results": dict(sorted(product_results.items())),
            "rejection_results": dict(sorted(rejection_results.items())),
            "legacy_hp5_result": {
                "returncode": legacy.returncode,
                "output_sha256": _sha(legacy_output),
                "portable_normalized_recipe_sha256": _normalized_recipe(
                    legacy_recipe, portable_paths=True
                ),
            },
            "gates": gates,
            "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
            "decision": (
                "Enforce the authoritative available product-look catalog at the "
                "safe-rich-product-v1 CLI boundary; preserve historical profiles."
            ),
            "claim_ceiling": config["claim_ceiling"],
        }
    finally:
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)
    report["owned_runtime_residue_zero"] = not WORK_ROOT.exists()
    if not report["owned_runtime_residue_zero"]:
        report["status"] = "FAIL_CLOSED"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u7_2i_product_look_catalog_enforcement_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    order = tuple(config["available_product_looks"])
    if args.order == "reverse":
        order = tuple(reversed(order))
    report = build_report(config_path=args.config, order=order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
