from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.product_desktop import (
    ProductDesktopWorkflow,
    _safe_output_stem,
)
from src.inference.render_contract import atomic_write_json, sha256_file
from src.inference.style_safe_engine import replay_style_safe_recipe_to_file

CONFIG = ROOT / "configs/u7_11a_desktop_single_look_batch_v1.json"
CONTRACT = ROOT / "docs/planning/U7_11A_DESKTOP_SINGLE_LOOK_BATCH_CONTRACT.md"
CORE = ROOT / "src/inference/product_desktop.py"
UI = ROOT / "src/inference/product_desktop_ui.py"
TEST = ROOT / "tests/test_u7_11a_desktop_single_look_batch.py"
AUDIT_TEST = ROOT / "tests/test_audit_u7_11a_desktop_single_look_batch.py"
SCRATCH_PARENT = ROOT / "tmp"
IMPLEMENTATION_COMMIT = "41e1c633f4e6be157ed3cad5178c36f3cac82d79"


def _source(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:43, :61]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + offset) % 251,
            (xx * 5 + yy * 13 + 17 + offset * 2) % 251,
            (xx * 7 + yy * 19 + 41 + offset * 3) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
    }


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(recipe))
    normalized["input"]["path"] = f"<INPUT:{Path(recipe['input']['path']).name}>"
    normalized["output"]["path"] = f"<OUTPUT:{Path(recipe['output']['path']).name}>"
    return normalized


def _normalized_receipt(
    receipt: dict[str, Any], recipe_semantics: dict[str, str]
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(receipt))
    normalized.pop("batch_id", None)
    for row in normalized["jobs"]:
        row["recipe_sha256"] = recipe_semantics[row["recipe_path"]]
    return normalized


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip().lower()


def _remove_case_root(case_root: Path) -> None:
    resolved = case_root.resolve(strict=True)
    parent = SCRATCH_PARENT.resolve(strict=True)
    if resolved.parent != parent or not resolved.name.startswith("u7-11a-case-"):
        raise RuntimeError("refusing to clean an unowned formal root")
    shutil.rmtree(resolved)


def _run_case(case_root: Path, selection_order: str) -> dict[str, Any]:
    case_root.mkdir()
    first = case_root / "B strange SOURCE.png"
    second = case_root / "a-source.png"
    _source(first, 1)
    _source(second, 2)
    selected = (first, second) if selection_order == "forward" else (second, first)
    scratch = case_root / "scratch"
    scratch.mkdir()
    workflow = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=3_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
    )
    destination = case_root / "published-batch"
    replay_paths: list[Path] = []
    try:
        state, bound = workflow.render_batch_previews(selected, 0.625)
        progress: list[list[Any]] = []
        batch = workflow.export_batch(
            bound,
            "ektar_100",
            destination,
            progress=lambda done, total, name: progress.append([done, total, name]),
        )
        batch_payloads: dict[str, bytes] = {
            path.name: path.read_bytes() for path in sorted(destination.iterdir())
        }
        batch_json_sha256 = sha256_file(destination / "batch.json")
        raw_receipt_sha256 = _canonical_sha256(batch.receipt)
        batch_recipes: dict[str, dict[str, Any]] = {
            row["recipe_path"]: json.loads(
                (destination / row["recipe_path"]).read_text("utf-8")
            )
            for row in batch.receipt["jobs"]
        }
        recipe_semantics = {
            name: _canonical_sha256(_normalized_recipe(recipe))
            for name, recipe in batch_recipes.items()
        }
        normalized_receipt = _normalized_receipt(batch.receipt, recipe_semantics)
        shutil.rmtree(destination)
        destination.mkdir()

        children: list[dict[str, Any]] = []
        for index, source in enumerate(bound, 1):
            stem = f"{index:04d}-{_safe_output_stem(source.path)}-ektar_100"
            image_name = f"{stem}.png"
            recipe_name = f"{stem}.recipe.json"
            output = destination / image_name
            completed = subprocess.run(
                list(
                    workflow._build_export_command(
                        source.path,
                        "ektar_100",
                        output,
                        0.625,
                    )
                ),
                cwd=ROOT,
                env=_environment(),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr or completed.stdout)
            recipe_path = destination / recipe_name
            recipe = batch_recipes[recipe_name]
            replay = case_root / f"replay-{image_name}"
            replay_paths.append(replay)
            replay_sha256 = replay_style_safe_recipe_to_file(
                recipe,
                profile_path=ROOT
                / "configs/render_profiles/safe_rich_product_v1.json",
                output_path=replay,
                root=ROOT,
            )
            children.append(
                {
                    "direct_image_exact": output.read_bytes()
                    == batch_payloads[image_name],
                    "direct_recipe_exact": recipe_path.read_bytes()
                    == batch_payloads[recipe_name],
                    "input_basename": source.basename,
                    "input_sha256": source.sha256,
                    "output_name": image_name,
                    "output_sha256": sha256_file(output),
                    "recipe_name": recipe_name,
                    "recipe_sha256": hashlib.sha256(
                        batch_payloads[recipe_name]
                    ).hexdigest(),
                    "recipe_semantics_sha256": recipe_semantics[recipe_name],
                    "replay_exact": replay.read_bytes() == output.read_bytes(),
                    "replay_sha256": replay_sha256,
                }
            )
        direct_member_count = len(tuple(destination.iterdir()))
        absolute_path_disclosed = str(case_root.resolve()) in json.dumps(
            batch.receipt, sort_keys=True
        )
        workflow_cleaned = workflow.close()
        preview_residue_count = len(tuple(scratch.iterdir()))
        return {
            "absolute_path_disclosed_in_receipt": absolute_path_disclosed,
            "batch_id": batch.batch_id,
            "batch_json_sha256": batch_json_sha256,
            "children": children,
            "direct_member_count": direct_member_count,
            "input_order": [row.basename for row in bound],
            "job_count": batch.job_count,
            "look_amount": state.look_amount,
            "normalized_receipt_sha256": _canonical_sha256(normalized_receipt),
            "raw_receipt_sha256": raw_receipt_sha256,
            "progress": progress,
            "receipt_claim": normalized_receipt["claim"],
            "receipt_paths_relative": all(
                not Path(row[key]).is_absolute()
                for row in normalized_receipt["jobs"]
                for key in ("output_path", "recipe_path")
            ),
            "style_id": batch.style_id,
            "workflow_cleaned": workflow_cleaned,
            "preview_residue_count": preview_residue_count,
        }
    finally:
        workflow.close()
        for path in replay_paths:
            path.unlink(missing_ok=True)
        if destination.is_dir():
            shutil.rmtree(destination)
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)
        if scratch.is_dir() and not tuple(scratch.iterdir()):
            scratch.rmdir()


def _targeted_test_once() -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(TEST),
        ],
        cwd=ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    match = re.search(r"(?m)^(\d+) passed", completed.stdout)
    return {
        "passed_count": int(match.group(1)) if match else 0,
        "returncode": completed.returncode,
        "stderr_empty": not completed.stderr,
    }


def _targeted_tests() -> dict[str, Any]:
    runs = [_targeted_test_once(), _targeted_test_once()]
    return {
        "all_pass": all(
            row["returncode"] == 0
            and row["passed_count"] == 21
            and row["stderr_empty"]
            for row in runs
        ),
        "runs": runs,
    }


def build_report(order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    config = json.loads(CONFIG.read_text("utf-8"))
    execution_commit = _git_head()
    baseline = {path.name for path in SCRATCH_PARENT.glob("u7-11a-case-*")}
    with tempfile.TemporaryDirectory(
        prefix="u7-11a-formal-controller-", dir=SCRATCH_PARENT
    ) as controller_raw:
        controller = Path(controller_raw)
        case_root = SCRATCH_PARENT / "u7-11a-case-fixed"
        if os.path.lexists(case_root):
            raise RuntimeError("formal case root must be absent")
        primary = _run_case(case_root, order)
        _remove_case_root(case_root)
        confirmation = _run_case(
            case_root, "reverse" if order == "forward" else "forward"
        )
        _remove_case_root(case_root)
        targeted = _targeted_tests()
        controller_residue_count = len(tuple(controller.iterdir()))
    residue_zero = {
        path.name for path in SCRATCH_PARENT.glob("u7-11a-case-*")
    } == baseline
    same_scientific_result = primary == confirmation
    children = primary["children"]
    gates = {
        "all_direct_cli_images_exact": all(
            row["direct_image_exact"] for row in children
        ),
        "all_direct_cli_recipes_exact": all(
            row["direct_recipe_exact"] for row in children
        ),
        "all_replays_exact": all(row["replay_exact"] for row in children),
        "canonical_forward_reverse_exact": same_scientific_result,
        "raw_child_and_receipt_identities_exact": same_scientific_result
        and bool(primary["batch_id"])
        and bool(primary["batch_json_sha256"])
        and bool(primary["raw_receipt_sha256"])
        and all(bool(row["recipe_sha256"]) for row in children),
        "claim_ceiling_exact": primary["receipt_claim"]
        == {
            "calibrated_stock_response": False,
            "evidence_grade": "look-approximation",
            "output_label": "film-inspired / Look Approximation",
            "physical_film_reproduction": False,
            "stock_distinguishability": False,
        },
        "exact_member_counts": primary["job_count"] == 2
        and primary["direct_member_count"] == 4,
        "explicit_single_look_and_amount": primary["style_id"] == "ektar_100"
        and primary["look_amount"] == 0.625,
        "progress_exact": primary["progress"]
        == [
            [1, 2, primary["input_order"][0]],
            [2, 2, primary["input_order"][1]],
        ],
        "receipt_is_path_bounded": primary["receipt_paths_relative"]
        and not primary["absolute_path_disclosed_in_receipt"],
        "repo_relative_p_storage": str(SCRATCH_PARENT.resolve()).casefold().startswith(
            "p:\\neuro_film_storage\\tmp"
        ),
        "targeted_core_ui_controls_pass": targeted["all_pass"],
        "workflow_and_formal_residue_zero": primary["workflow_cleaned"]
        and primary["preview_residue_count"] == 0
        and controller_residue_count == 0
        and residue_zero,
    }
    scientific = {
        "canonical_case": primary,
        "claim_ceiling": config["claim_ceiling"],
        "gates": gates,
        "selection_order_confirmation_exact": same_scientific_result,
        "storage": {
            "repo_relative_root": "tmp",
            "resolved_root": "P:/neuro_film_storage/tmp",
            "windows_directory_publication": os.name == "nt",
        },
        "targeted_tests": targeted,
    }
    if _git_head() != execution_commit:
        raise RuntimeError("source commit changed during formal execution")
    passed = all(gates.values())
    return {
        "schema": "kmcfm.u7-11a-desktop-single-look-batch-result.v1",
        "status": (
            "PASS_PRIVATE_U7_11A_DESKTOP_SINGLE_LOOK_BATCH"
            if passed
            else "FAIL_CLOSED_U7_11A_DESKTOP_SINGLE_LOOK_BATCH"
        ),
        "source_commit": execution_commit,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "bindings": {
            "audit_test_sha256": sha256_file(AUDIT_TEST),
            "config_sha256": sha256_file(CONFIG),
            "contract_sha256": sha256_file(CONTRACT),
            "core_sha256": sha256_file(CORE),
            "test_sha256": sha256_file(TEST),
            "ui_sha256": sha256_file(UI),
        },
        "scientific": scientific,
        "scientific_identity_sha256": _canonical_sha256(scientific),
        "claim": (
            "Private Windows desktop single-look batch mechanics only; every named "
            "look remains film-inspired / Look Approximation. No calibrated stock "
            "response, physical-film reproduction, stock distinguishability, public "
            "installer, arbitrary media, or cross-platform GUI claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(strict=False)
    if os.path.lexists(output):
        raise FileExistsError(f"output must be absent: {output}")
    report = build_report(args.order)
    atomic_write_json(output, report)
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
