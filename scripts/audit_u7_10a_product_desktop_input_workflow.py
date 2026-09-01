from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageGrab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.film_physics.create_only_file import (
    PublishedFileIdentity,
    publish_create_only,
    remove_if_published,
)
from src.inference.product_desktop import PRODUCT_LOOKS, ProductDesktopWorkflow
from src.inference.product_desktop_ui import build_product_desktop_app
from src.inference.render_contract import atomic_write_json, sha256_file

CONFIG = ROOT / "configs/u7_10a_product_desktop_input_workflow_v1.json"
CONTRACT = ROOT / "docs/planning/U7_10A_PRODUCT_DESKTOP_INPUT_WORKFLOW_CONTRACT.md"
CORE = ROOT / "src/inference/product_desktop.py"
UI = ROOT / "src/inference/product_desktop_ui.py"
ENTRYPOINT = ROOT / "scripts/open_product_desktop.py"
TEST = ROOT / "tests/test_u7_10a_product_desktop_input_workflow.py"
SCRATCH_PARENT = ROOT / "tmp"
IMPLEMENTATION_COMMIT = "080c116dbf1683397508d712aa4f94fd7011e15f"


def _source(path: Path, *, width: int, height: int, offset: int) -> None:
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + offset) % 251,
            (xx * 5 + yy * 13 + 17 + offset * 3) % 251,
            (xx * 7 + yy * 19 + 41 + offset * 5) % 251,
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


def _normalized_recipe_sha256(recipe: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(recipe))
    payload["input"]["path"] = "<INPUT>"
    payload["output"]["path"] = "<OUTPUT>"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _case(
    root: Path, name: str, geometry: tuple[int, int], amount: float
) -> dict[str, Any]:
    source = root / f"{name}.png"
    width, height = geometry
    _source(source, width=width, height=height, offset=len(name) * 7)
    scratch = root / f"{name}-scratch"
    scratch.mkdir()
    workflow = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=4_000,
        tile_size=23,
        tile_workers=1,
        png_compression=6,
    )
    state = workflow.render_previews(source, amount)
    preview_rows = [
        {
            "output_sha256": row["output_sha256"],
            "style_id": row["style_id"],
        }
        for row in state.manifest["rows"]
    ]
    exports: list[dict[str, Any]] = []
    for look in PRODUCT_LOOKS:
        style = look["style_id"]
        output = root / f"{name}-{style}.png"
        command = workflow.export_command(style, output)
        direct = subprocess.run(
            command,
            cwd=ROOT,
            env=_environment(),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if direct.returncode != 0:
            raise RuntimeError(direct.stderr or direct.stdout)
        recipe_path = output.with_suffix(".recipe.json")
        direct_image = output.read_bytes()
        direct_recipe = recipe_path.read_bytes()
        output.unlink()
        recipe_path.unlink()
        receipt = workflow.export(style, output)
        exports.append(
            {
                "claim": receipt.recipe["claim"]["evidence_grade"],
                "direct_cli_image_exact": output.read_bytes() == direct_image,
                "direct_cli_recipe_exact": recipe_path.read_bytes() == direct_recipe,
                "output_bit_depth": receipt.recipe["output"]["bit_depth"],
                "output_sha256": receipt.output_sha256,
                "recipe_semantics_sha256": _normalized_recipe_sha256(receipt.recipe),
                "source_commit": receipt.recipe["software"]["commit"],
                "style_id": style,
            }
        )
        output.unlink()
        recipe_path.unlink()
    preview_cleaned = workflow.close()
    scratch_residue = tuple(scratch.iterdir())
    source.unlink()
    scratch.rmdir()
    return {
        "amount": amount,
        "exports": exports,
        "geometry": [width, height],
        "input_sha256": state.input_sha256,
        "name": name,
        "preview_count": len(preview_rows),
        "preview_distinct_count": len({row["output_sha256"] for row in preview_rows}),
        "preview_pixels": state.manifest["preview_pixels"],
        "preview_rows": preview_rows,
        "preview_workspace_cleaned": preview_cleaned,
        "scratch_residue_count": len(scratch_residue),
    }


def _capture_visual(source: Path, scratch: Path, output: Path) -> int:
    root = tk.Tk()
    workflow = ProductDesktopWorkflow(root=ROOT, scratch_root=scratch)
    app = build_product_desktop_app(root, workflow, initial_input=source)
    try:
        root.update()
        state = workflow.render_previews(source, 0.65)
        app._preview_complete(state)
        root.update()
        ImageGrab.grab(window=root.winfo_id()).save(
            output, format="PNG", compress_level=6
        )
    finally:
        app.close()
    return 0


def _gui_visual(root: Path) -> tuple[dict[str, Any], bytes]:
    source = root / "gui-source.png"
    _source(source, width=61, height=43, offset=29)
    scratch = root / "gui-scratch"
    scratch.mkdir()
    screenshot = root / "gui-ready-state.png"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--capture-visual",
            str(source),
            str(scratch),
            str(screenshot),
        ],
        cwd=ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    residue = tuple(scratch.iterdir())
    screenshot_payload = screenshot.read_bytes() if screenshot.is_file() else b""
    if screenshot_payload:
        with Image.open(screenshot) as opened:
            screenshot_size = list(opened.size)
    else:
        screenshot_size = [0, 0]
    screenshot.unlink(missing_ok=True)
    source.unlink()
    scratch.rmdir()
    return (
        {
            "ready_state_png_sha256": hashlib.sha256(screenshot_payload).hexdigest(),
            "ready_state_size": screenshot_size,
            "returncode": completed.returncode,
            "scratch_residue_count": len(residue),
            "stderr_empty": not completed.stderr,
        },
        screenshot_payload,
    )


def _scientific_identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_report(order: str, *, visual_output: Path | None = None) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cases = [
        ("landscape", (83, 57), 0.35),
        ("portrait", (49, 79), 0.8),
    ]
    if order == "reverse":
        cases.reverse()
    execution_commit = _git_head()
    with tempfile.TemporaryDirectory(
        prefix="u7-10a-formal-", dir=SCRATCH_PARENT
    ) as temporary:
        root = Path(temporary)
        rows = [_case(root, name, geometry, amount) for name, geometry, amount in cases]
        gui, visual_payload = _gui_visual(root)
        final_residue_count = len(tuple(root.iterdir()))
    rows.sort(key=lambda row: row["name"])
    all_exports = [export for row in rows for export in row["exports"]]
    gates = {
        "all_claims_look_approximation": all(
            row["claim"] == "look-approximation" for row in all_exports
        ),
        "all_direct_cli_images_exact": all(
            row["direct_cli_image_exact"] for row in all_exports
        ),
        "all_direct_cli_recipes_exact": all(
            row["direct_cli_recipe_exact"] for row in all_exports
        ),
        "all_final_outputs_rgb16": all(
            row["output_bit_depth"] == 16 for row in all_exports
        ),
        "all_preview_counts_exact": all(row["preview_count"] == 3 for row in rows),
        "all_preview_sets_distinct": all(
            row["preview_distinct_count"] == 3 for row in rows
        ),
        "all_previews_bounded": all(row["preview_pixels"] <= 4_000 for row in rows),
        "all_session_commits_exact": all(
            row["source_commit"] == execution_commit for row in all_exports
        ),
        "final_owned_residue_zero": final_residue_count == 0,
        "gui_smoke_pass": gui["returncode"] == 0
        and gui["scratch_residue_count"] == 0
        and gui["stderr_empty"]
        and gui["ready_state_size"][0] >= 920
        and gui["ready_state_size"][1] >= 680
        and bool(visual_payload),
        "preview_workspace_cleanup_pass": all(
            row["preview_workspace_cleaned"] and row["scratch_residue_count"] == 0
            for row in rows
        ),
    }
    scientific = {
        "cases": rows,
        "claim_ceiling": config["claim_ceiling"],
        "gates": gates,
        "gui_smoke": gui,
    }
    if _git_head() != execution_commit:
        raise RuntimeError("source commit changed during formal execution")
    passed = all(gates.values())
    if visual_output is not None:
        destination = visual_output.resolve(strict=False)
        if destination.exists():
            raise FileExistsError(f"visual output must be absent: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = destination.with_name(f".{destination.name}.{os.getpid()}.stage")
        stage.write_bytes(visual_payload)
        details = stage.lstat()
        stage_identity = PublishedFileIdentity(
            path=stage,
            device=details.st_dev,
            inode=details.st_ino,
        )
        try:
            publish_create_only(stage, destination)
        finally:
            if stage.exists():
                remove_if_published(stage_identity)
    return {
        "schema": "kmcfm.u7-10a-product-desktop-input-workflow-result.v1",
        "status": (
            "PASS_PRIVATE_U7_10A_PRODUCT_DESKTOP_INPUT_WORKFLOW"
            if passed
            else "FAIL_CLOSED_U7_10A_PRODUCT_DESKTOP_INPUT_WORKFLOW"
        ),
        "source_commit": execution_commit,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "bindings": {
            "config_sha256": sha256_file(CONFIG),
            "contract_sha256": sha256_file(CONTRACT),
            "core_sha256": sha256_file(CORE),
            "entrypoint_sha256": sha256_file(ENTRYPOINT),
            "test_sha256": sha256_file(TEST),
            "ui_sha256": sha256_file(UI),
        },
        "scientific": scientific,
        "scientific_identity_sha256": _scientific_identity(scientific),
        "claim": (
            "Private repo-local Windows new-input Look Approximation workflow mechanics "
            "only; no calibrated stock, physical-film, public installer, arbitrary "
            "platform, population preference, or stock-distinguishability claim."
        ),
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visual-output", type=Path, default=None)
    parser.add_argument(
        "--capture-visual",
        nargs=3,
        metavar=("SOURCE", "SCRATCH", "OUTPUT"),
    )
    args = parser.parse_args()
    if args.capture_visual is not None:
        source, scratch, visual = (Path(value) for value in args.capture_visual)
        return _capture_visual(source, scratch, visual)
    if args.order is None or args.output is None:
        parser.error("--order and --output are required for a formal report")
    output = args.output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"output must be absent: {output}")
    report = build_report(args.order, visual_output=args.visual_output)
    atomic_write_json(output, report)
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
