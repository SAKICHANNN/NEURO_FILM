from __future__ import annotations

import json
import os
import shutil
import sys
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import TclError

import numpy as np
import pytest
from PIL import Image

from src.inference.product_desktop import ProductDesktopWorkflow
from src.inference.product_desktop_ui import (
    _batch_recovery_workspace,
    build_product_desktop_app,
)

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:32, :48]
    rgb = np.stack(
        (
            (xx * 13 + yy * 7 + offset * 29) % 251,
            (xx * 3 + yy * 17 + offset * 47 + 19) % 251,
            (xx * 11 + yy * 5 + offset * 61 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _sources(tmp_path: Path) -> tuple[Path, ...]:
    rows = []
    for index in range(3):
        source = tmp_path / f"u7-12e-{index:02d}.png"
        _source(source, index)
        rows.append(source)
    return tuple(rows)


def _workflow(tmp_path: Path, suffix: str) -> ProductDesktopWorkflow:
    scratch = tmp_path / f"scratch-{suffix}"
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


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _pump(root: tk.Tk, predicate: Callable[[], bool], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        root.update()
        time.sleep(0.01)
    assert predicate()


def _root() -> tk.Tk:
    try:
        root = tk.Tk()
    except TclError as exc:
        pytest.skip(f"Tk is unavailable: {exc}")
    root.withdraw()
    return root


def test_contract_freezes_recovery_ui_route_and_claim_ceiling() -> None:
    contract = json.loads(
        (ROOT / "configs/u7_12e_desktop_batch_recovery_ui_v1.json").read_text(
            "utf-8"
        )
    )
    assert contract["ui"]["batch_button_copy"].startswith("Export / resume")
    assert contract["ui"]["legacy_export_batch_called_by_ui"] is False
    assert contract["ui"]["paused_receipt_is_error"] is False
    assert contract["owned_paths"]["parent_core_fixture_output_or_evidence_changes"] is False
    assert contract["claim_ceiling"]["private_windows_python_ui_integration_only"]
    assert contract["claim_ceiling"]["calibrated_stock_response"] is False


def test_recovery_workspace_is_normalized_sibling_and_destination_bound(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "Batch Folder"
    equivalent = tmp_path / "." / "Batch Folder"
    first = _batch_recovery_workspace(destination)
    second = _batch_recovery_workspace(equivalent)
    other = _batch_recovery_workspace(tmp_path / "Other Batch")
    assert first == second
    assert first.parent == destination.resolve(strict=False).parent
    assert first.name.startswith(".kmcfm-u7-12e-")
    assert first.name.endswith(".recovery")
    assert len(first.name) == len(".kmcfm-u7-12e-") + 32 + len(".recovery")
    assert first != other
    if os.name == "nt":
        assert first == _batch_recovery_workspace(
            Path(str(destination).swapcase())
        )


@pytest.mark.skipif(os.name != "nt", reason="formal recovery publication is Windows-only")
def test_ui_pauses_then_fresh_session_resumes_exact_u7_11a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path)
    destination = tmp_path / "published-batch"
    workspace = _batch_recovery_workspace(destination)
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.filedialog.asksaveasfilename",
        lambda **_kwargs: str(destination),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showerror",
        lambda title, message: shown.append((title, message)),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showinfo",
        lambda title, message: shown.append((title, message)),
    )

    workflow = _workflow(tmp_path, "pause")
    root = _root()
    app = build_product_desktop_app(root, workflow, initial_input=sources[0])
    try:
        app._set_inputs(sources)
        state, bound = workflow.render_batch_previews(sources, 0.625)
        app._batch_preview_complete((state, bound))
        assert app.export_button.cget("text") == "Export / resume 3 PNG16 + recipes"
        app.look_buttons["portra_400"].invoke()
        original_progress = app._batch_progress

        def pause_after_one(completed: int, total: int, basename: str) -> None:
            original_progress(completed, total, basename)
            if completed == 1:
                app._batch_cancel.set()

        app._batch_progress = pause_after_one
        app.export()
        _pump(root, lambda: app._batch_thread is None)
        assert workspace.is_dir()
        assert not destination.exists()
        assert app.preview_ready and not app.busy and not app.batch_active
        assert app.status.get().startswith("Batch paused: 1/3 complete")
        assert "0 reused, 1 new" in app.status.get()
        assert "same photos, Look, strength and destination" in app.status.get()
        assert shown == []
    finally:
        app.close()

    resumed_workflow = _workflow(tmp_path, "resume")
    resumed_root = _root()
    resumed_app = build_product_desktop_app(
        resumed_root, resumed_workflow, initial_input=sources[0]
    )
    try:
        resumed_app._set_inputs(sources)
        state, rebound = resumed_workflow.render_batch_previews(sources, 0.625)
        resumed_app._batch_preview_complete((state, rebound))
        resumed_app.look_buttons["portra_400"].invoke()
        resumed_app.export()
        _pump(resumed_root, lambda: resumed_app._batch_thread is None)
        assert destination.is_dir()
        assert not workspace.exists()
        assert resumed_app.status.get() == "Batch complete: 3 photos in published-batch"
        assert shown[-1][0] == "Batch export complete"
        recovered_bytes = _tree_bytes(destination)
        recovered_receipt = json.loads((destination / "batch.json").read_text("utf-8"))
    finally:
        resumed_app.close()

    shutil.rmtree(destination)
    legacy_workflow = _workflow(tmp_path, "legacy")
    _, legacy_bound = legacy_workflow.render_batch_previews(sources, 0.625)
    legacy = legacy_workflow.export_batch(
        legacy_bound,
        "portra_400",
        destination,
    )
    assert legacy.receipt == recovered_receipt
    assert _tree_bytes(destination) == recovered_bytes
    assert legacy_workflow.close()
