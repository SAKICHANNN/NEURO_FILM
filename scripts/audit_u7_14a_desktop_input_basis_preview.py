#!/usr/bin/env python3
"""Committed-head formal audit for U7.14A input-basis desktop preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import three_stock_preview as preview_module
from src.inference.product_desktop import ProductDesktopError, ProductDesktopWorkflow
from src.inference.product_desktop_ui import build_product_desktop_app
from src.inference.three_stock_preview import (
    preview_dimensions,
    render_three_stock_previews_to_directory,
)
from src.preprocess import (
    load_jpeg_preview_working_image,
    load_raw_preview_working_image,
    save_srgb8,
    working_image_to_srgb_float,
)

CONFIG_PATH = ROOT / "configs/u7_14a_desktop_input_basis_preview_v1.json"
BOUND_PATHS = (
    "configs/u7_14a_desktop_input_basis_preview_v1.json",
    "docs/planning/U7_14A_DESKTOP_INPUT_BASIS_PREVIEW_CONTRACT.md",
    "src/inference/three_stock_preview.py",
    "src/inference/product_desktop.py",
    "src/inference/product_desktop_ui.py",
    "tests/test_u7_14a_desktop_input_basis_preview.py",
    "scripts/audit_u7_14a_desktop_input_basis_preview.py",
    "tests/test_u7_14a_desktop_input_basis_preview_audit.py",
)
STYLE_IDS = ("velvia_50", "portra_400", "ektar_100")
FORMAL_ROOT = ROOT / "tmp/u7_14a_desktop_input_basis_preview_formal"


class U714AError(RuntimeError):
    """Raised when the frozen U7.14A audit cannot execute safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _git_blob_sha256(path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True, capture_output=True
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        value = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3 or value.size == 0:
        raise U714AError(f"invalid RGB8 preview: {path}")
    return np.ascontiguousarray(value)


def _portable_manifest(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    portable = json.loads(json.dumps(manifest, sort_keys=True))
    resolved_root = root.resolve()
    for row in portable["rows"]:
        output = Path(row["output_path"])
        if output.parent != resolved_root:
            raise U714AError("preview row escaped its owned output root")
        row["output_path"] = f"<root>/{output.name}"
    if "input_preview" in portable:
        output = Path(portable["input_preview"]["output_path"])
        if output.parent != resolved_root:
            raise U714AError("input preview escaped its owned output root")
        portable["input_preview"]["output_path"] = f"<root>/{output.name}"
    return portable


def _source_row(config: dict[str, Any], source_id: str) -> dict[str, Any]:
    for row in config["sources"]:
        if row["source_id"] == source_id:
            return row
    raise U714AError(f"unknown source id: {source_id}")


def _verify_source(row: dict[str, Any]) -> Path:
    path = ROOT / row["path"]
    if (
        not path.is_file()
        or path.stat().st_size != int(row["bytes"])
        or sha256_file(path) != row["sha256"]
    ):
        raise U714AError(f"source identity mismatch: {row['source_id']}")
    return path


def _render_kwargs(config: dict[str, Any], source: Path) -> dict[str, Any]:
    render = config["render"]
    suffix = source.suffix.casefold()
    return {
        "root": ROOT,
        "profile_path": ROOT / "configs/render_profiles/safe_rich_v1.json",
        "statistics_path": ROOT / "configs/film_color_stats.json",
        "guardrails_path": ROOT / "configs/color_guardrails.json",
        "max_preview_pixels": int(render["max_preview_pixels"]),
        "max_preview_width": int(render["max_preview_width"]),
        "max_preview_height": int(render["max_preview_height"]),
        "look_amount": float(render["look_amount"]),
        "seed": int(render["seed"]),
        "tile_size": int(render["tile_size"]),
        "tile_workers": int(render["tile_workers"]),
        "png_compression": int(render["png_compression"]),
        "jpeg_scaled_decode": suffix in {".jpg", ".jpeg"},
        "raw_half_size_decode": suffix == ".dng",
    }


def _oracle_input_preview(
    config: dict[str, Any], row: dict[str, Any], source: Path, output: Path
) -> tuple[int, int]:
    width, height = (
        preview_dimensions(
            int(row.get("source_width", 0) or 1),
            int(row.get("source_height", 0) or 1),
            int(config["render"]["max_preview_pixels"]),
            max_width=int(config["render"]["max_preview_width"]),
            max_height=int(config["render"]["max_preview_height"]),
        )
        if "source_width" in row
        else tuple(row["preview_dimensions"])
    )
    if source.suffix.casefold() in {".jpg", ".jpeg"}:
        working = load_jpeg_preview_working_image(
            source, target_width=int(width), target_height=int(height)
        )
    else:
        working = load_raw_preview_working_image(source)
    resized = np.ascontiguousarray(
        cv2.resize(
            working.pixels,
            (int(width), int(height)),
            interpolation=cv2.INTER_AREA,
        ),
        dtype=np.float32,
    )
    display = working_image_to_srgb_float(replace(working, pixels=resized))
    save_srgb8(
        display,
        output,
        png_compression=int(config["render"]["png_compression"]),
    )
    return int(width), int(height)


def _counted_candidate_render(
    config: dict[str, Any], source: Path, destination: Path
) -> tuple[dict[str, Any], int]:
    suffix = source.suffix.casefold()
    attribute = (
        "load_jpeg_preview_working_image"
        if suffix in {".jpg", ".jpeg"}
        else "load_raw_preview_working_image"
    )
    original: Callable[..., Any] = getattr(preview_module, attribute)
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    setattr(preview_module, attribute, counted)
    try:
        manifest = render_three_stock_previews_to_directory(
            source,
            destination,
            include_input_preview=True,
            **_render_kwargs(config, source),
        )
    finally:
        setattr(preview_module, attribute, original)
    return manifest, calls


def _workflow_facts(config: dict[str, Any], source: Path, root: Path) -> dict[str, Any]:
    scratch = root / "workflow"
    scratch.mkdir()
    workflow = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        tile_size=int(config["render"]["tile_size"]),
        tile_workers=int(config["render"]["tile_workers"]),
        png_compression=int(config["render"]["png_compression"]),
    )
    state = workflow.render_previews(source, float(config["render"]["look_amount"]))
    bound = workflow.input_preview_bytes()
    input_row = state.manifest["input_preview"]
    facts = {
        "bound_sha256": hashlib.sha256(bound).hexdigest(),
        "manifest_sha256": input_row["output_sha256"],
        "three_look_count": len(workflow.preview_bytes()),
        "close_clean": workflow.close(),
    }
    if any(scratch.iterdir()):
        raise U714AError("workflow scratch residue remained")
    if facts["bound_sha256"] != facts["manifest_sha256"]:
        raise U714AError("workflow input-preview identity drifted")

    tamper_scratch = root / "workflow-tamper"
    tamper_scratch.mkdir()
    tamper = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=tamper_scratch,
        python_executable=Path(sys.executable),
        tile_size=int(config["render"]["tile_size"]),
        tile_workers=int(config["render"]["tile_workers"]),
        png_compression=int(config["render"]["png_compression"]),
    )
    tamper_state = tamper.render_previews(
        source, float(config["render"]["look_amount"])
    )
    tamper_path = Path(tamper_state.manifest["input_preview"]["output_path"])
    tamper_path.write_bytes(b"u7-14a-foreign-tamper")
    rejected = False
    try:
        tamper.input_preview_bytes()
    except ProductDesktopError:
        rejected = True
    preserved = (
        not tamper.close() and tamper_path.read_bytes() == b"u7-14a-foreign-tamper"
    )
    facts["tamper_rejected"] = rejected
    facts["foreign_tamper_preserved"] = preserved
    # The auditor injected and owns this control value; product cleanup correctly
    # refuses ownership, so the audit removes only its own enclosing temp root.
    shutil.rmtree(tamper_scratch)
    return facts


def _ui_facts(config: dict[str, Any], source: Path, root: Path) -> dict[str, Any]:
    import tkinter as tk

    scratch = root / "ui"
    scratch.mkdir()
    workflow = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        tile_size=int(config["render"]["tile_size"]),
        tile_workers=int(config["render"]["tile_workers"]),
        png_compression=int(config["render"]["png_compression"]),
    )
    window = tk.Tk()
    window.withdraw()
    app = build_product_desktop_app(window, workflow, initial_input=source)
    try:
        state = workflow.render_previews(source, float(config["render"]["look_amount"]))
        app._preview_complete(state)
        shown = app.input_preview_image is not None
        caption = str(app.input_preview_caption.cget("text"))
        no_selection = app.style.get() == ""
        export_disabled = str(app.export_button.cget("state")) == "disabled"
        look_ids = tuple(app.look_buttons)
        app.amount.set(0.5)
        app._amount_changed()
        cleared = (
            app.input_preview_image is None
            and app.input_preview_label.cget("text") == "Input basis not rendered"
            and workflow.preview_state is None
        )
    finally:
        if window.winfo_exists():
            app.close()
    if any(scratch.iterdir()):
        raise U714AError("UI scratch residue remained")
    return {
        "shown": shown,
        "caption": caption,
        "no_selection": no_selection,
        "export_disabled": export_disabled,
        "look_ids": list(look_ids),
        "cleared_after_amount_change": cleared,
    }


def _source_facts(
    config: dict[str, Any], row: dict[str, Any], source: Path, root: Path
) -> dict[str, Any]:
    source_before = sha256_file(source)
    omitted_root = root / "omitted"
    false_root = root / "false"
    candidate_root = root / "candidate"
    oracle_path = root / "oracle.png"
    omitted = render_three_stock_previews_to_directory(
        source, omitted_root, **_render_kwargs(config, source)
    )
    explicit_false = render_three_stock_previews_to_directory(
        source,
        false_root,
        include_input_preview=False,
        **_render_kwargs(config, source),
    )
    candidate, decode_calls = _counted_candidate_render(config, source, candidate_root)
    expected_width, expected_height = tuple(row["preview_dimensions"])
    _oracle_input_preview(config, row, source, oracle_path)

    omitted_names = sorted(path.name for path in omitted_root.iterdir())
    false_names = sorted(path.name for path in false_root.iterdir())
    look_hashes = {
        style_id: sha256_file(candidate_root / f"{style_id}.preview.png")
        for style_id in STYLE_IDS
    }
    baseline_hashes = {
        style_id: sha256_file(omitted_root / f"{style_id}.preview.png")
        for style_id in STYLE_IDS
    }
    input_path = candidate_root / "input.preview.png"
    input_rgb = _rgb8(input_path)
    input_row = candidate.get("input_preview")
    expected_input_row = {
        "role": "input_basis",
        "output_path": str(input_path.resolve()),
        "output_sha256": sha256_file(input_path),
        "source_kind": row["source_kind"],
        "display_adapter": "existing WorkingImage to display-sRGB adapter",
        "claim": "generic display adapter, not a calibrated camera rendering",
    }
    facts = {
        "source_id": row["source_id"],
        "source_sha256": source_before,
        "default_manifest_exact": _portable_manifest(omitted, omitted_root)
        == _portable_manifest(explicit_false, false_root),
        "default_tree_exact": omitted_names == false_names,
        "default_member_names": omitted_names,
        "candidate_decode_calls": decode_calls,
        "look_hashes": look_hashes,
        "baseline_look_hashes": baseline_hashes,
        "look_bytes_exact": look_hashes == baseline_hashes,
        "input_manifest_exact": input_row == expected_input_row,
        "input_file_sha256": sha256_file(input_path),
        "oracle_file_sha256": sha256_file(oracle_path),
        "input_oracle_exact": input_path.read_bytes() == oracle_path.read_bytes(),
        "input_width": int(input_rgb.shape[1]),
        "input_height": int(input_rgb.shape[0]),
        "geometry_exact": (int(input_rgb.shape[1]), int(input_rgb.shape[0]))
        == (int(expected_width), int(expected_height)),
        "finite_rgb8": bool(np.isfinite(input_rgb).all()),
        "style_ids": [item["style_id"] for item in candidate["rows"]],
        "workflow": _workflow_facts(config, source, root),
        "ui": _ui_facts(config, source, root),
        "source_immutable": sha256_file(source) == source_before,
    }
    return facts


def build_report(config_path: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise U714AError("order must be forward or reverse")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema"] != "kmcfm.u7-14a-desktop-input-basis-preview-contract.v1":
        raise U714AError("config schema mismatch")
    if _git_output("status", "--short", "--untracked-files=no"):
        raise U714AError("tracked worktree must be clean")
    head = _git_output("rev-parse", "HEAD")
    rows = list(config["sources"])
    execution_rows = rows if order == "forward" else list(reversed(rows))
    if FORMAL_ROOT.exists():
        raise U714AError("formal root must be absent")
    FORMAL_ROOT.mkdir(parents=True)
    try:
        source_facts = []
        for row in execution_rows:
            source = _verify_source(row)
            row_root = FORMAL_ROOT / row["source_id"]
            row_root.mkdir()
            source_facts.append(_source_facts(config, row, source, row_root))
        source_facts.sort(key=lambda item: item["source_id"])
        gates = {
            "tracked_clean": True,
            "source_identities_exact": all(
                item["source_immutable"] for item in source_facts
            ),
            "default_manifest_and_tree_exact": all(
                item["default_manifest_exact"]
                and item["default_tree_exact"]
                and item["default_member_names"]
                == [
                    "ektar_100.preview.png",
                    "portra_400.preview.png",
                    "preview.json",
                    "velvia_50.preview.png",
                ]
                for item in source_facts
            ),
            "one_selected_source_decode": all(
                item["candidate_decode_calls"] == 1 for item in source_facts
            ),
            "three_look_files_exact": all(
                item["look_bytes_exact"] for item in source_facts
            ),
            "input_preview_oracle_exact": all(
                item["input_manifest_exact"] and item["input_oracle_exact"]
                for item in source_facts
            ),
            "input_preview_geometry_and_pixels": all(
                item["geometry_exact"] and item["finite_rgb8"] for item in source_facts
            ),
            "catalog_role_unchanged": all(
                item["style_ids"] == list(STYLE_IDS) for item in source_facts
            ),
            "workflow_binding_and_tamper": all(
                item["workflow"]["three_look_count"] == 3
                and item["workflow"]["close_clean"]
                and item["workflow"]["tamper_rejected"]
                and item["workflow"]["foreign_tamper_preserved"]
                for item in source_facts
            ),
            "ui_truthful_and_clears": all(
                item["ui"]["shown"]
                and "not a calibrated camera rendering" in item["ui"]["caption"]
                and item["ui"]["no_selection"]
                and item["ui"]["export_disabled"]
                and item["ui"]["look_ids"] == list(STYLE_IDS)
                and item["ui"]["cleared_after_amount_change"]
                for item in source_facts
            ),
        }
        scientific = {
            "schema": "kmcfm.u7-14a-desktop-input-basis-preview-science.v1",
            "sources": source_facts,
            "gates": gates,
            "claim": config["claim_ceiling"],
        }
        report = {
            "schema": "kmcfm.u7-14a-desktop-input-basis-preview-report.v1",
            "status": (
                "PASS_PRIVATE_U7_14A_DESKTOP_INPUT_BASIS_PREVIEW"
                if all(gates.values())
                else "FAIL_CLOSED_U7_14A_DESKTOP_INPUT_BASIS_PREVIEW"
            ),
            "producer_commit": head,
            "config_sha256": sha256_file(config_path),
            "bound_git_blob_sha256": {
                path: _git_blob_sha256(path) for path in BOUND_PATHS
            },
            "scientific_identity": _canonical_sha256(scientific),
            "scientific": scientific,
            "owned_residue_zero": False,
        }
    finally:
        shutil.rmtree(FORMAL_ROOT, ignore_errors=False)
    report["owned_residue_zero"] = not FORMAL_ROOT.exists()
    report["scientific"]["gates"]["owned_residue_zero"] = report["owned_residue_zero"]
    report["status"] = (
        "PASS_PRIVATE_U7_14A_DESKTOP_INPUT_BASIS_PREVIEW"
        if all(report["scientific"]["gates"].values())
        else "FAIL_CLOSED_U7_14A_DESKTOP_INPUT_BASIS_PREVIEW"
    )
    report["scientific_identity"] = _canonical_sha256(report["scientific"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve(strict=False)
    if report_path.exists() or not report_path.parent.is_dir():
        raise U714AError("report destination must be absent with an existing parent")
    report = build_report(args.config.resolve(strict=True), args.order)
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"].startswith("PASS_PRIVATE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
