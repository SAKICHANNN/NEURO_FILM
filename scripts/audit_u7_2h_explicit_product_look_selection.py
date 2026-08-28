"""Audit explicit manual selection for the product look profile."""

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalized_recipe_sha256(path: Path) -> str:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    recipe["software"]["commit"] = "<normalized>"
    return _canonical_sha256(recipe)


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


def _message(stderr: str, needle: str) -> str:
    return next((line.strip() for line in stderr.splitlines() if needle in line), "")


def build_report(*, config_path: Path, order: tuple[str, ...]) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    WORK_ROOT.mkdir(parents=True)
    source = WORK_ROOT / "source.png"
    _source(source)
    source_before = _sha256(source)
    results: dict[str, dict[str, Any]] = {}
    omitted_message = ""
    blocked_message = ""
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
            results[style] = {
                "returncode": completed.returncode,
                "output_sha256": _sha256(output) if output.exists() else None,
                "normalized_recipe_sha256": (
                    _normalized_recipe_sha256(recipe) if recipe.exists() else None
                ),
                "recipe_commit": (
                    json.loads(recipe.read_text(encoding="utf-8"))["software"]["commit"]
                    if recipe.exists()
                    else None
                ),
            }

        legacy = WORK_ROOT / "legacy_default.png"
        legacy_run = _run(
            source,
            legacy,
            (
                "--use-render-profile",
                "--render-profile",
                str(LEGACY_PROFILE),
                "--write-recipe",
            ),
        )
        legacy_recipe = legacy.with_suffix(".recipe.json")

        omitted_output = WORK_ROOT / "omitted_product.png"
        omitted = _run(
            WORK_ROOT / "missing_input.png",
            omitted_output,
            (
                "--use-render-profile",
                "--render-profile",
                str(PRODUCT_PROFILE),
                "--write-recipe",
            ),
        )
        omitted_message = _message(
            omitted.stderr, "requires an explicit --style product look selection"
        )

        blocked_output = WORK_ROOT / "blocked_generic.png"
        blocked = _run(
            source,
            blocked_output,
            (
                "--style",
                "generic_bw",
                "--use-render-profile",
                "--render-profile",
                str(PRODUCT_PROFILE),
            ),
        )
        blocked_message = _message(blocked.stderr, "product look 'generic_bw' is unavailable")

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        oracle = config["prechange_oracle"]
        product_exact = all(
            results[style]["returncode"] == 0
            and results[style]["output_sha256"]
            == oracle["product"][style]["output_sha256"]
            and results[style]["normalized_recipe_sha256"]
            == oracle["product"][style]["normalized_recipe_sha256"]
            and results[style]["recipe_commit"] == head
            for style in config["available_product_looks"]
        )
        legacy_exact = (
            legacy_run.returncode == 0
            and _sha256(legacy) == oracle["legacy_default"]["output_sha256"]
            and _normalized_recipe_sha256(legacy_recipe)
            == oracle["legacy_default"]["normalized_recipe_sha256"]
            and json.loads(legacy_recipe.read_text(encoding="utf-8"))["software"][
                "commit"
            ]
            == head
        )
        rejection_residue = any(
            path.exists()
            for path in (
                omitted_output,
                omitted_output.with_suffix(".recipe.json"),
                blocked_output,
                blocked_output.with_suffix(".recipe.json"),
            )
        )
        gates = {
            "fixture_source_exact": source_before == oracle["fixture_sha256"],
            "source_immutable": _sha256(source) == source_before,
            "parent_evidence_exact": all(
                _sha256(ROOT / row["path"]) == row["sha256"]
                for row in config["parent_evidence"]
            ),
            "profiles_exact": (
                _sha256(PRODUCT_PROFILE) == config["source_locks"]["product_profile_sha256"]
                and _sha256(LEGACY_PROFILE)
                == config["source_locks"]["legacy_profile_sha256"]
            ),
            "omitted_product_style_rejects": omitted.returncode != 0
            and bool(omitted_message)
            and "missing_input" not in omitted.stderr,
            "rejection_residue_zero": not rejection_residue,
            "three_explicit_product_looks_exact": product_exact,
            "legacy_default_exact": legacy_exact,
            "blocked_generic_bw_unchanged": blocked.returncode != 0
            and "3/16" in blocked_message,
        }
        report = {
            "schema_id": "neuro-film.u7-2h-explicit-product-look-selection-result.v1",
            "experiment_id": "U7.2H",
            "implementation_commit": head,
            "bindings": {
                "config": {
                    "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": _sha256(config_path),
                },
                "cli": {"path": "scripts/render_film.py", "sha256": _sha256(SCRIPT)},
                "product_profile": {
                    "path": str(PRODUCT_PROFILE.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": _sha256(PRODUCT_PROFILE),
                },
                "legacy_profile": {
                    "path": str(LEGACY_PROFILE.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": _sha256(LEGACY_PROFILE),
                },
            },
            "product_results": dict(sorted(results.items())),
            "legacy_result": {
                "returncode": legacy_run.returncode,
                "output_sha256": _sha256(legacy),
                "normalized_recipe_sha256": _normalized_recipe_sha256(legacy_recipe),
            },
            "rejection_messages": {
                "omitted_product_style": omitted_message,
                "blocked_generic_bw": blocked_message,
            },
            "gates": gates,
            "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
            "decision": (
                "Require explicit caller selection for safe-rich-product-v1 while "
                "preserving the historical Velvia default outside the product profile."
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
        default=ROOT / "configs/u7_2h_explicit_product_look_selection_v1.json",
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
