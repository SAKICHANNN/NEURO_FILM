#!/usr/bin/env python3
"""Audit the U7.15C basename-only batch representative repair."""

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

from src.inference.product_desktop import ProductDesktopWorkflow, sha256_file
from src.inference.product_desktop_ui import build_product_desktop_app

CONTRACT = ROOT / "configs/u7_15c_desktop_batch_representative_path_disclosure_v1.json"
PARENT_UI_PATH = "src/inference/product_desktop_ui.py"
AUTOMATIC = "Automatic · canonical first"
EXPECTED_PARENT_HEAD = "020bb364d9d86824b8af4bb22418bb7a1e27c9ea"

BINDINGS = (
    "configs/u7_15c_desktop_batch_representative_path_disclosure_v1.json",
    "docs/planning/U7_15C_DESKTOP_BATCH_REPRESENTATIVE_PATH_DISCLOSURE_CONTRACT.md",
    "scripts/audit_u7_15c_desktop_batch_representative_path_disclosure.py",
    "src/inference/product_desktop.py",
    "src/inference/product_desktop_ui.py",
    "tests/test_u7_15c_desktop_batch_representative_path_disclosure.py",
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _source(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:31, :47]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + offset) % 251,
            (xx * 5 + yy * 13 + 17 + offset * 2) % 251,
            (xx * 7 + yy * 19 + 41 + offset * 3) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _workflow(root: Path, name: str) -> ProductDesktopWorkflow:
    scratch = root / name
    scratch.mkdir()
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=2_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _source_role(value: Path, first: Path, second: Path) -> str:
    resolved = value.resolve(strict=True)
    if resolved == first.resolve(strict=True):
        return "private-alpha/same-name.png"
    if resolved == second.resolve(strict=True):
        return "private-beta/same-name.png"
    raise RuntimeError("selected source escaped the frozen two-role cohort")


def _ui_observation(
    root_path: Path, first: Path, second: Path
) -> dict[str, Any]:
    import tkinter as tk

    workflow = _workflow(root_path, "ui-scratch")
    root = tk.Tk()
    root.withdraw()
    app = build_product_desktop_app(root, workflow)
    try:
        app._set_inputs((second, first))
        values = tuple(str(value) for value in app.representative_combo.cget("values"))
        app.amount.set(0.625)
        snapped = app._snap_visible_amount()
        app.representative.set(values[1])
        selected_first = app._selected_representative_path()
        app._representative_changed()
        first_text = app.input_text.get()
        app.representative.set(values[2])
        selected_second = app._selected_representative_path()
        app._representative_changed()
        second_text = app.input_text.get()
        visible = (*values, first_text, second_text, app.status.get())
        return {
            "values": list(values),
            "selected_001": _source_role(selected_first, first, second),
            "selected_002": _source_role(selected_second, first, second),
            "input_texts": [first_text, second_text],
            "visible_strength_percent": int(app.amount_label.cget("text")[:-1]),
            "visible_strength_amount": snapped,
            "parent_paths_hidden": all(
                str(parent) not in text
                for text in visible
                for parent in (first.parent, second.parent, root_path)
            ),
        }
    finally:
        if root.winfo_exists():
            app.close()


def _batch_arm(
    root_path: Path,
    name: str,
    sources: tuple[Path, Path],
    representative: Path | None,
    first: Path,
    second: Path,
) -> dict[str, Any]:
    workflow = _workflow(root_path, f"{name}-scratch")
    destination = root_path / "batch"
    try:
        state, bound = workflow.render_batch_previews(
            sources,
            0.63,
            representative_path=representative,
        )
        receipt = workflow.export_batch(bound, "portra_400", destination)
        return {
            "representative": _source_role(state.input_path, first, second),
            "canonical_inputs": [
                _source_role(row.path, first, second) for row in bound
            ],
            "receipt_sha256": receipt.receipt_sha256,
            "tree_hashes": _tree_hashes(destination),
        }
    finally:
        workflow.close()
        if destination.exists():
            shutil.rmtree(destination)


def evaluate(order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    if os.name != "nt":
        raise RuntimeError("U7.15C formal audit requires Windows")

    contract = json.loads(CONTRACT.read_text("utf-8"))
    if contract["parent_head"] != EXPECTED_PARENT_HEAD:
        raise RuntimeError("U7.15C parent drift")
    source_commit = _git("rev-parse", "HEAD")
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean")

    parent_source = _git("show", f"{EXPECTED_PARENT_HEAD}:{PARENT_UI_PATH}")
    parent_expression = 'label = f"{index:03d} · {path}"'
    current_expression = 'label = f"{index:03d} · {path.name}"'
    parent_disclosure = parent_expression in parent_source
    current_source = (ROOT / PARENT_UI_PATH).read_text("utf-8")

    owned_root = ROOT / "tmp" / "u7_15c_formal"
    owned_root.mkdir(parents=True)
    try:
        first = owned_root / "private-alpha" / "same-name.png"
        second = owned_root / "private-beta" / "same-name.png"
        _source(first, 1)
        _source(second, 2)
        source_hashes_before = {
            first.relative_to(owned_root).as_posix(): sha256_file(first),
            second.relative_to(owned_root).as_posix(): sha256_file(second),
        }
        ui = _ui_observation(owned_root, first, second)
        arm_names = ["default", "explicit"]
        if order == "reverse":
            arm_names.reverse()
        arms: dict[str, dict[str, Any]] = {}
        for name in arm_names:
            arms[name] = _batch_arm(
                owned_root,
                name,
                (second, first),
                None if name == "default" else second,
                first,
                second,
            )
        source_hashes_after = {
            first.relative_to(owned_root).as_posix(): sha256_file(first),
            second.relative_to(owned_root).as_posix(): sha256_file(second),
        }
        result = {
            "parent_disclosure_expression_present": parent_disclosure,
            "parent_example_exposes_first_parent": str(first.parent)
            in f"001 · {first}",
            "parent_example_exposes_second_parent": str(second.parent)
            in f"002 · {second}",
            "ui": ui,
            "arms": arms,
            "source_hashes": source_hashes_before,
            "source_immutable": source_hashes_before == source_hashes_after,
        }
    finally:
        if owned_root.exists():
            shutil.rmtree(owned_root)

    expected_values = [AUTOMATIC, "001 · same-name.png", "002 · same-name.png"]
    default = result["arms"]["default"]
    explicit = result["arms"]["explicit"]
    gates = {
        "parent_disclosure_reproduced": (
            bool(parent_disclosure)
            and bool(result["parent_example_exposes_first_parent"])
            and bool(result["parent_example_exposes_second_parent"])
        ),
        "current_expression_exact": (
            current_expression in current_source
            and parent_expression not in current_source
        ),
        "visible_values_exact": result["ui"]["values"] == expected_values,
        "visible_parent_paths_hidden": bool(result["ui"]["parent_paths_hidden"]),
        "duplicate_basename_selection_exact": (
            result["ui"]["selected_001"] == "private-beta/same-name.png"
            and result["ui"]["selected_002"] == "private-alpha/same-name.png"
        ),
        "explicit_representative_exact": (
            default["representative"] == "private-alpha/same-name.png"
            and explicit["representative"] == "private-beta/same-name.png"
        ),
        "canonical_order_exact": (
            default["canonical_inputs"]
            == explicit["canonical_inputs"]
            == ["private-alpha/same-name.png", "private-beta/same-name.png"]
        ),
        "batch_bytes_and_receipt_exact": (
            default["receipt_sha256"] == explicit["receipt_sha256"]
            and default["tree_hashes"] == explicit["tree_hashes"]
        ),
        "visible_strength_exact": (
            result["ui"]["visible_strength_percent"] == 63
            and result["ui"]["visible_strength_amount"] == 0.63
        ),
        "source_immutable": bool(result["source_immutable"]),
        "product_core_unchanged": (
            sha256_file(ROOT / "src/inference/product_desktop.py")
            == contract["parent_bindings"]["product_desktop"]["sha256"]
        ),
        "tracked_diff_clean": not bool(
            _git("status", "--short", "--untracked-files=no")
        ),
        "zero_owned_residue": not owned_root.exists(),
    }
    scientific = {
        "schema": "kmcfm.u7-15c-desktop-batch-representative-path-disclosure-scientific.v1",
        "result": result,
        "gates": gates,
    }
    stable_identity = _sha256_bytes(_json_bytes(scientific))
    return {
        "schema": "kmcfm.u7-15c-desktop-batch-representative-path-disclosure-report.v1",
        "status": (
            "PASS_PRIVATE_U7_15C_DESKTOP_BATCH_REPRESENTATIVE_PATH_DISCLOSURE"
            if all(gates.values())
            else "FAIL_CLOSED_U7_15C_DESKTOP_BATCH_REPRESENTATIVE_PATH_DISCLOSURE"
        ),
        "formal_source_commit": source_commit,
        "bindings": {path: sha256_file(ROOT / path) for path in BINDINGS},
        "scientific": scientific,
        "stable_identity": stable_identity,
        "claim": {
            "evidence_grade": "look-approximation",
            "private_windows_tk_presentation_only": True,
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "public_or_cross_platform_release": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.order)
    encoded = _json_bytes(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(encoded)
    print(json.dumps({"status": report["status"], "sha256": _sha256_bytes(encoded)}))
    return 0 if report["status"].startswith("PASS_PRIVATE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
