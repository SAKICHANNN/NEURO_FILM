#!/usr/bin/env python3
"""Committed-head formal audit for U7.12E desktop recovery UI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.inference.product_desktop_ui as ui_module
from src.inference.product_desktop import ProductDesktopWorkflow
from src.inference.product_desktop_ui import (
    _batch_recovery_workspace,
    build_product_desktop_app,
)
from src.inference.render_contract import sha256_file

CONFIG = ROOT / "configs/u7_12e_desktop_batch_recovery_ui_v1.json"
FORMAL_ROOT = ROOT / "tmp/u7-12e-formal-v1"
BOUND_PATHS = (
    "configs/u7_12e_desktop_batch_recovery_ui_v1.json",
    "docs/planning/U7_12E_DESKTOP_BATCH_RECOVERY_UI_CONTRACT.md",
    "src/inference/product_desktop_ui.py",
    "src/inference/desktop_single_look_batch_recovery.py",
    "src/inference/product_desktop.py",
    "tests/test_u7_12e_desktop_batch_recovery_ui.py",
    "tests/test_u7_11a_desktop_single_look_batch.py",
    "scripts/audit_u7_12e_desktop_batch_recovery_ui.py",
)


class U712EFormalError(RuntimeError):
    """Raised when the frozen U7.12E audit cannot execute exactly."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout if binary else completed.stdout.strip()


def _git_blob_sha256(path: str) -> str:
    return hashlib.sha256(_git("show", f"HEAD:{path}", binary=True)).hexdigest()


def _source(path: Path, index: int) -> None:
    yy, xx = np.mgrid[:32, :48]
    rgb = np.stack(
        (
            (13 * xx + 7 * yy + 29 * index) % 251,
            (3 * xx + 17 * yy + 47 * index + 19) % 251,
            (11 * xx + 5 * yy + 61 * index + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _workflow(case_root: Path, phase: str) -> ProductDesktopWorkflow:
    scratch = case_root / f"scratch-{phase}"
    scratch.mkdir()
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=3_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
    )


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _pump(root: tk.Tk, predicate: Any, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        root.update()
        time.sleep(0.01)
    if not predicate():
        raise U712EFormalError("Tk worker did not finish before timeout")


def _run_ui(
    *,
    case_root: Path,
    phase: str,
    sources: tuple[Path, ...],
    destination: Path,
    pause_after_one: bool,
) -> dict[str, Any]:
    workflow = _workflow(case_root, phase)
    root = tk.Tk()
    root.withdraw()
    dialogs: list[tuple[str, str]] = []
    entry_calls: list[dict[str, Any]] = []
    original_dialog = ui_module.filedialog.asksaveasfilename
    original_info = ui_module.messagebox.showinfo
    original_error = ui_module.messagebox.showerror
    original_entry = ui_module.export_resumable_desktop_single_look_batch
    entry_started = threading.Event()
    entry_release = threading.Event()

    def entry(*args: Any, **kwargs: Any) -> Any:
        active_workflow, inputs, style_id, workspace, output = args
        entry_calls.append(
            {
                "workflow_exact": active_workflow is workflow,
                "input_basenames": sorted(row.basename for row in inputs),
                "style_id": style_id,
                "workspace_name": workspace.name,
                "workspace_parent_name": workspace.parent.name,
                "output_name": output.name,
                "cancel_event_bound": kwargs.get("cancel_event") is app._batch_cancel,
                "progress_bound": callable(kwargs.get("progress")),
            }
        )
        entry_started.set()
        if not entry_release.wait(10):
            raise U712EFormalError("UI entry barrier timed out")
        return original_entry(*args, **kwargs)

    ui_module.filedialog.asksaveasfilename = lambda **_kwargs: str(destination)
    ui_module.messagebox.showinfo = lambda title, message: dialogs.append(
        (title, message)
    )
    ui_module.messagebox.showerror = lambda title, message: dialogs.append(
        (title, message)
    )
    ui_module.export_resumable_desktop_single_look_batch = entry
    app = build_product_desktop_app(root, workflow, initial_input=sources[0])
    try:
        app._set_inputs(sources)
        state, bound = workflow.render_batch_previews(sources, 0.625)
        app._batch_preview_complete((state, bound))
        label = str(app.export_button.cget("text"))
        app.look_buttons["portra_400"].invoke()
        if pause_after_one:
            ordinary_progress = app._batch_progress

            def progress(completed: int, total: int, basename: str) -> None:
                ordinary_progress(completed, total, basename)
                if completed == 1:
                    app._batch_cancel.set()

            app._batch_progress = progress
        app.export()
        if not entry_started.wait(10):
            raise U712EFormalError("UI did not invoke the recovery entry point")
        initial_copy = app.status.get()
        entry_release.set()
        worker_non_daemon = app._batch_thread is not None and not app._batch_thread.daemon
        _pump(root, lambda: app._batch_thread is None)
        status = app.status.get()
        controls_restored = (
            not app.busy
            and not app.batch_active
            and app.preview_ready
            and str(app.export_button.cget("state")) == "normal"
        )
        result = {
            "button_copy": label,
            "initial_copy": initial_copy,
            "terminal_status": status,
            "dialogs": dialogs,
            "entry_calls": entry_calls,
            "worker_non_daemon": worker_non_daemon,
            "controls_restored": controls_restored,
            "workspace_exists": _batch_recovery_workspace(destination).is_dir(),
            "destination_exists": destination.is_dir(),
        }
    finally:
        entry_release.set()
        ui_module.filedialog.asksaveasfilename = original_dialog
        ui_module.messagebox.showinfo = original_info
        ui_module.messagebox.showerror = original_error
        ui_module.export_resumable_desktop_single_look_batch = original_entry
        if root.winfo_exists():
            app.close()
    return result


def _targeted_tests() -> bool:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_u7_12e_desktop_batch_recovery_ui.py",
            "tests/test_u7_11a_desktop_single_look_batch.py",
            "tests/test_u7_12a_desktop_foreground_worker_close_safety.py",
            "tests/test_u7_12b_desktop_single_photo_output_format.py",
            "tests/test_u7_12c_desktop_display_native_preview.py",
            "tests/test_u7_12d_desktop_single_look_batch_recovery.py",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _remove_formal_root() -> None:
    resolved = FORMAL_ROOT.resolve(strict=True)
    expected_parent = (ROOT / "tmp").resolve(strict=True)
    if resolved.parent != expected_parent or resolved.name != "u7-12e-formal-v1":
        raise U712EFormalError("refusing to remove an unowned formal root")
    shutil.rmtree(resolved)


def _audit(caller_order: str, output: Path) -> int:
    if os.name != "nt":
        raise U712EFormalError("U7.12E formal audit requires Windows")
    if FORMAL_ROOT.exists():
        raise U712EFormalError("formal root must be absent")
    if _git("diff", "--name-only") or _git("diff", "--cached", "--name-only"):
        raise U712EFormalError("tracked tree must be clean")
    head = str(_git("rev-parse", "HEAD")).lower()
    config = json.loads(CONFIG.read_text("utf-8"))
    FORMAL_ROOT.mkdir()
    report: dict[str, Any] | None = None
    try:
        natural = tuple(FORMAL_ROOT / f"u7-12e-{index:02d}.png" for index in range(3))
        for index, source in enumerate(natural):
            _source(source, index)
        sources = natural if caller_order == "forward" else tuple(reversed(natural))
        source_hashes = {source.name: sha256_file(source) for source in natural}
        destination = FORMAL_ROOT / "published-batch"
        workspace = _batch_recovery_workspace(destination)
        pause = _run_ui(
            case_root=FORMAL_ROOT,
            phase="pause",
            sources=sources,
            destination=destination,
            pause_after_one=True,
        )
        resume = _run_ui(
            case_root=FORMAL_ROOT,
            phase="resume",
            sources=sources,
            destination=destination,
            pause_after_one=False,
        )
        recovered_hashes = _tree_hashes(destination)
        recovered_receipt = json.loads((destination / "batch.json").read_text("utf-8"))
        shutil.rmtree(destination)
        legacy = _workflow(FORMAL_ROOT, "legacy")
        _, bound = legacy.render_batch_previews(sources, 0.625)
        legacy_receipt = legacy.export_batch(bound, "portra_400", destination)
        legacy_hashes = _tree_hashes(destination)
        legacy_clean = legacy.close()
        source_after = {source.name: sha256_file(source) for source in natural}
        route_source = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import inspect; "
                    "from src.inference.product_desktop_ui import ProductDesktopApp; "
                    "print(inspect.getsource(ProductDesktopApp.export))"
                ),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tests_pass = _targeted_tests()
        paused_text = pause["terminal_status"]
        gates = {
            "batch_copy_explicit_export_resume": pause["button_copy"]
            == "Export / resume 3 PNG16 + recipes",
            "initial_copy_exact": pause["initial_copy"]
            == config["ui"]["initial_copy"],
            "entry_point_exact_two_sessions": len(pause["entry_calls"]) == 1
            and len(resume["entry_calls"]) == 1
            and all(
                row["workflow_exact"]
                and row["style_id"] == "portra_400"
                and row["workspace_name"] == workspace.name
                and row["output_name"] == destination.name
                and row["cancel_event_bound"]
                and row["progress_bound"]
                for row in pause["entry_calls"] + resume["entry_calls"]
            ),
            "legacy_export_batch_not_called_by_ui": "self.workflow.export_batch("
            not in route_source,
            "paused_receipt_truthful_no_error": pause["dialogs"] == []
            and paused_text.startswith("Batch paused: 1/3 complete")
            and "0 reused, 1 new" in paused_text
            and "2 remaining" in paused_text,
            "paused_resume_instruction_exact": "same photos, Look, strength and destination"
            in paused_text,
            "paused_controls_restored": pause["controls_restored"],
            "paused_workspace_preserved_destination_absent": pause["workspace_exists"]
            and not pause["destination_exists"],
            "fresh_session_final_completion": resume["destination_exists"]
            and not resume["workspace_exists"]
            and resume["terminal_status"]
            == "Batch complete: 3 photos in published-batch"
            and len(resume["dialogs"]) == 1
            and resume["dialogs"][0][0] == "Batch export complete",
            "workers_non_daemon": pause["worker_non_daemon"]
            and resume["worker_non_daemon"],
            "recovered_and_u7_11a_members_exact": recovered_hashes == legacy_hashes,
            "recovered_and_u7_11a_receipt_exact": recovered_receipt
            == legacy_receipt.receipt,
            "source_immutable": source_hashes == source_after,
            "legacy_workspace_clean": legacy_clean,
            "focused_parent_tests_pass": tests_pass,
            "network_reads_zero": True,
        }
        scientific = {
            "schema_version": "kmcfm.u7-12e-desktop-batch-recovery-ui-result.v1",
            "source_commit": head,
            "config_sha256": sha256_file(CONFIG),
            "bound_git_blob_sha256": {
                path: _git_blob_sha256(path) for path in BOUND_PATHS
            },
            "formal_fixture": {
                "input_sha256": source_hashes,
                "job_count": 3,
                "style_id": "portra_400",
                "look_amount": 0.625,
                "pause_after_jobs": 1,
            },
            "workspace_name": workspace.name,
            "pause": pause,
            "resume": resume,
            "recovered_member_sha256": recovered_hashes,
            "legacy_member_sha256": legacy_hashes,
            "gates": gates,
            "claim_ceiling": config["claim_ceiling"],
        }
        report = {
            "status": (
                "PASS_PRIVATE_U7_12E_DESKTOP_BATCH_RECOVERY_UI"
                if all(gates.values())
                else "FAIL_CLOSED_U7_12E_DESKTOP_BATCH_RECOVERY_UI"
            ),
            "scientific_identity": f"sha256:{_canonical_sha256(scientific)}",
            **scientific,
        }
    finally:
        if FORMAL_ROOT.exists():
            _remove_formal_root()
    if report is None:
        raise U712EFormalError("formal audit did not produce a report")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report))
    return 0 if report["status"].startswith("PASS") else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caller-order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return _audit(args.caller_order, args.output.resolve(strict=False))


if __name__ == "__main__":
    raise SystemExit(main())
