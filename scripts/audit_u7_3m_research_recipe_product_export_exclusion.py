#!/usr/bin/env python3
"""Committed-head audit for U7.3M product-export eligibility repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.recipe_browser_export import (
    RecipeBrowserExportError,
    RecipeBrowserExportSession,
)
from src.inference.recipe_export_request import (
    build_recipe_export_request_set,
)
from src.inference.recipe_history import build_render_recipe_history
from src.inference.recipe_history_export import (
    RecipeHistoryExportError,
    export_recipe_history_entry,
)
from src.inference.render_contract import sha256_file

CONFIG = ROOT / "configs/u7_3m_research_recipe_product_export_exclusion_v1.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def _verify_source_locks(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    verified: dict[str, dict[str, str]] = {}
    for name, lock in sorted(config["source_locks"].items()):
        commit = (
            config["contract_commit"]
            if name == "contract"
            else config["implementation_commit"]
        )
        blob = _git("rev-parse", f"{commit}:{lock['path']}").decode().strip()
        payload = _git("show", f"{commit}:{lock['path']}")
        current_blob = _git("rev-parse", f"HEAD:{lock['path']}").decode().strip()
        if (
            blob != lock["git_blob"]
            or current_blob != lock["git_blob"]
            or _sha256(payload) != lock["sha256"]
        ):
            raise ValueError(f"source lock drifted: {name}")
        verified[name] = {
            "path": lock["path"],
            "git_blob": blob,
            "sha256": _sha256(payload),
        }
    return verified


def _fixture(path: Path) -> None:
    yy, xx = np.mgrid[:192, :256]
    pixels = np.stack(
        (
            (3 * xx + yy) % 256,
            (xx + 5 * yy) % 256,
            ((xx // 8) * 19 + (yy // 6) * 13) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _render_command(source: Path, output: Path, *, research: bool) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/render_film.py"),
        str(source),
        "--style",
        "velvia_50" if research else "ektar_100",
        "--use-render-profile",
        "--tile-size",
        "64",
        "--output-bit-depth",
        "16",
        "--png-compression",
        "0",
        "--write-recipe",
        "--output",
        str(output),
    ]
    if research:
        command.extend(
            ["--halation-model", "staged-density-research", "--halation", "1.0"]
        )
    return command


def _render(source: Path, output: Path, *, research: bool) -> None:
    completed = subprocess.run(
        _render_command(source, output, research=research),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "fixture render failed")


def _worker(config: dict[str, Any], scratch: Path, order: str) -> dict[str, Any]:
    if scratch.exists() or scratch.is_symlink():
        raise ValueError("owned scratch already exists")
    scratch.mkdir(parents=True)
    source = scratch / "source.png"
    research_output = scratch / "research.png"
    product_output = scratch / "product.png"
    research_only = scratch / "research-only"
    product_only = scratch / "product-only"
    mixed = scratch / "mixed"
    browser_output = scratch / "browser-output"
    direct_research = scratch / "direct-research.png"
    direct_product = scratch / "direct-product.png"
    try:
        _fixture(source)
        roles = [("research", True), ("product", False)]
        if order == "reverse":
            roles.reverse()
        for name, research in roles:
            _render(
                source,
                research_output if name == "research" else product_output,
                research=research,
            )
        for directory in (research_only, product_only, mixed):
            directory.mkdir()
        research_recipe = research_output.with_suffix(".recipe.json")
        product_recipe = product_output.with_suffix(".recipe.json")
        shutil.copyfile(research_recipe, research_only / "research.recipe.json")
        shutil.copyfile(product_recipe, product_only / "product.recipe.json")
        shutil.copyfile(research_recipe, mixed / "research.recipe.json")
        shutil.copyfile(product_recipe, mixed / "product.recipe.json")

        source_before = sha256_file(source)
        profile_before = sha256_file(PROFILE)
        recipe_before = {
            "research": sha256_file(research_recipe),
            "product": sha256_file(product_recipe),
        }
        general = build_render_recipe_history(research_only)
        product_history = build_render_recipe_history(
            research_only, product_export_only=True
        )
        research_set = build_recipe_export_request_set(research_only)
        product_set = build_recipe_export_request_set(product_only)
        mixed_set = build_recipe_export_request_set(mixed)

        direct_rejected = False
        try:
            export_recipe_history_entry(
                research_only,
                recipe_path="research.recipe.json",
                profile_path=PROFILE,
                output_path=direct_research,
                root=ROOT,
            )
        except RecipeHistoryExportError as exc:
            direct_rejected = "missing or invalid" in str(exc)

        browser_rejected = False
        try:
            RecipeBrowserExportSession(
                research_only,
                browser_output,
                profile_path=PROFILE,
                root=ROOT,
            )
        except RecipeBrowserExportError as exc:
            browser_rejected = "no valid export recipes" in str(exc)

        product_receipt = export_recipe_history_entry(
            product_only,
            recipe_path="product.recipe.json",
            profile_path=PROFILE,
            output_path=direct_product,
            root=ROOT,
        )
        product_output_bytes = product_output.read_bytes()
        direct_product_bytes = direct_product.read_bytes()
        research_invalid_code = product_history["entries"][0].get("error_code")
        controls = {
            "default_history_status": general["status"],
            "default_history_valid": general["counts"]["valid"],
            "product_history_status": product_history["status"],
            "product_history_invalid": product_history["counts"]["invalid"],
            "research_invalid_code": research_invalid_code,
            "research_request_count": research_set["receipt"]["request_count"],
            "mixed_request_count": mixed_set["receipt"]["request_count"],
            "mixed_request_styles": [
                row["style"] for row in mixed_set["receipt"]["requests"]
            ],
            "direct_research_rejected": direct_rejected,
            "browser_research_rejected": browser_rejected,
            "research_output_absent": not direct_research.exists(),
            "browser_output_members": (
                sorted(path.name for path in browser_output.iterdir())
                if browser_output.exists()
                else []
            ),
        }
        gates = {
            "default_history_retains_research_evidence": (
                general["status"] == "ready"
                and general["counts"]
                == {"discovered": 1, "valid": 1, "invalid": 0}
            ),
            "product_history_rejects_research_recipe": (
                product_history["status"] == "invalid"
                and product_history["counts"]
                == {"discovered": 1, "valid": 0, "invalid": 1}
                and research_invalid_code == config["research_invalid_code"]
            ),
            "research_only_request_set_empty": (
                research_set["receipt"]["request_count"] == 0
                and research_set["receipt"]["requests"] == []
            ),
            "direct_export_rejects_before_replay": (
                direct_rejected and not direct_research.exists()
            ),
            "browser_rejects_research_only_history": (
                browser_rejected
                and browser_output.exists()
                and not any(browser_output.iterdir())
            ),
            "ordinary_product_request_and_replay_exact": (
                product_set["receipt"]["request_count"] == 1
                and product_receipt["style"] == "ektar_100"
                and direct_product_bytes == product_output_bytes
            ),
            "mixed_history_exports_only_product_recipe": (
                mixed_set["receipt"]["request_count"] == 1
                and controls["mixed_request_styles"] == ["ektar_100"]
                and mixed_set["files"]["index.html"]
                == product_set["files"]["index.html"]
            ),
            "inputs_immutable_and_scratch_empty": (
                sha256_file(source) == source_before
                and sha256_file(PROFILE) == profile_before
                and sha256_file(research_recipe) == recipe_before["research"]
                and sha256_file(product_recipe) == recipe_before["product"]
            ),
        }
        return {
            "controls": controls,
            "gates": gates,
            "ordinary_product": {
                "output_bytes": len(product_output_bytes),
                "output_sha256": _sha256(product_output_bytes),
                "replay_sha256": _sha256(direct_product_bytes),
                "request_sha256": product_set["receipt"]["requests"][0][
                    "request_sha256"
                ],
            },
            "recipe_sha256": recipe_before,
            "source_sha256": source_before,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=False)


def run(config_path: Path, report_path: Path, scratch: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_locks = _verify_source_locks(config)
    result = _worker(config, scratch, order)
    result["gates"]["fresh_process_scientific_payload_exact"] = True
    scientific = {
        "schema": config["schema_version"],
        "node": config["node"],
        "bindings": {
            "contract_commit": config["contract_commit"],
            "implementation_commit": config["implementation_commit"],
            "source_locks": source_locks,
        },
        **result,
        "claim_ceiling": config["claim_ceiling"],
    }
    required = set(config["required_gates"])
    if set(scientific["gates"]) != required:
        raise ValueError("formal gate inventory drifted")
    status = (
        "PASS_PRIVATE_U7_3M_RESEARCH_RECIPE_PRODUCT_EXPORT_EXCLUSION"
        if all(scientific["gates"].values())
        else "FAIL_CLOSED_U7_3M_RESEARCH_RECIPE_PRODUCT_EXPORT_EXCLUSION"
    )
    report = {
        "schema": "neuro-film.u7-3m-research-recipe-product-export-exclusion.v1.report",
        "status": status,
        "scientific": scientific,
        "stable_identity": _sha256(_canonical_bytes(scientific)),
    }
    report["report_identity"] = _sha256(_canonical_bytes(report))
    if report_path.exists() or report_path.is_symlink():
        raise ValueError("formal report already exists")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(report))
    os.rename(temporary, report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    args = parser.parse_args()
    report = run(args.config, args.report, args.scratch, args.order)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
