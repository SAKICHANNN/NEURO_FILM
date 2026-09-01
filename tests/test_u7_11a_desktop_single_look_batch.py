from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from tkinter import TclError
from typing import Any

import numpy as np
import pytest
from PIL import Image

import src.inference.product_desktop as desktop_module
from src.inference.product_desktop import (
    DesktopBatchReceipt,
    ProductDesktopError,
    ProductDesktopWorkflow,
)
from src.inference.product_desktop_ui import build_product_desktop_app
from src.inference.render_contract import sha256_file, verify_render_recipe_files
from src.inference.style_safe_engine import replay_style_safe_recipe_to_file

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path, offset: int = 0) -> None:
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


def _workflow(tmp_path: Path, **kwargs: object) -> ProductDesktopWorkflow:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=3_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
        **kwargs,  # type: ignore[arg-type]
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "B strange SOURCE.png"
    second = tmp_path / "a-source.png"
    _source(first, 1)
    _source(second, 2)
    return first, second


def _direct_pair(
    workflow: ProductDesktopWorkflow,
    source: Path,
    style_id: str,
    amount: float,
    output: Path,
) -> tuple[bytes, bytes]:
    completed = subprocess.run(
        list(workflow._build_export_command(source, style_id, output, amount)),
        cwd=ROOT,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("PYTHON")
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    recipe = output.with_suffix(".recipe.json")
    payload = (output.read_bytes(), recipe.read_bytes())
    output.unlink()
    recipe.unlink()
    return payload


def test_contract_freezes_single_look_atomic_batch_and_claim_ceiling() -> None:
    contract = json.loads(
        (ROOT / "configs/u7_11a_desktop_single_look_batch_v1.json").read_text(
            "utf-8"
        )
    )
    assert contract["limits"] == {
        "minimum_inputs": 1,
        "maximum_inputs": 100,
        "look_amount_minimum": 0.0,
        "look_amount_maximum": 1.0,
        "output_bit_depth": 16,
        "png_compression": 6,
        "tile_size": 256,
        "tile_workers": 1,
    }
    assert contract["batch_contract"]["single_explicit_look"] is True
    assert contract["batch_contract"]["publish_unit"] == "one create-only directory"
    assert contract["claim_ceiling"]["calibrated_stock_response"] is False
    assert contract["claim_ceiling"]["physical_film_reproduction"] is False
    assert contract["claim_ceiling"]["stock_distinguishability"] is False


def test_bind_batch_inputs_is_canonical_hash_bound_and_unique(tmp_path: Path) -> None:
    first, second = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    bound = workflow.bind_batch_inputs((first, second))
    assert [row.path for row in bound] == sorted(
        (first.resolve(), second.resolve()),
        key=lambda path: os.path.normcase(os.path.abspath(os.fspath(path))),
    )
    assert [row.sha256 for row in bound] == [sha256_file(row.path) for row in bound]
    assert all(row.size == row.path.stat().st_size for row in bound)
    with pytest.raises(ProductDesktopError, match="unique paths"):
        workflow.bind_batch_inputs((first, first))
    with pytest.raises(ProductDesktopError, match="one and 100"):
        workflow.bind_batch_inputs((first,) * 101)


def test_bind_batch_inputs_rejects_hardlink_alias(tmp_path: Path) -> None:
    first, _ = _inputs(tmp_path)
    alias = tmp_path / "alias.png"
    try:
        os.link(first, alias)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    workflow = _workflow(tmp_path)
    with pytest.raises(ProductDesktopError, match="unique files"):
        workflow.bind_batch_inputs((first, alias))


def test_preview_bytes_is_not_batch_platform_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _ = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    workflow.render_previews(first, 0.5)
    monkeypatch.setattr(
        desktop_module, "_batch_directory_rename_supported", lambda: False
    )
    assert set(workflow.preview_bytes()) == {"velvia_50", "portra_400", "ektar_100"}
    workflow.close()


def test_non_windows_batch_gate_precedes_stage_and_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _inputs(tmp_path)
    calls = 0

    def forbidden(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess([], 1, "", "forbidden")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    _, bound = workflow.render_batch_previews((first, second), 0.5)
    monkeypatch.setattr(
        desktop_module, "_batch_directory_rename_supported", lambda: False
    )
    destination = tmp_path / "batch"
    with pytest.raises(ProductDesktopError, match="requires Windows"):
        workflow.export_batch(bound, "ektar_100", destination)
    assert calls == 0
    assert not destination.exists()
    assert not list(tmp_path.glob(".batch.u7-11a-*.stage"))
    workflow.close()


@pytest.mark.skipif(os.name != "nt", reason="formal directory publication is Windows-only")
def test_two_input_batch_matches_direct_cli_and_strict_replay(tmp_path: Path) -> None:
    first, second = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    state, bound = workflow.render_batch_previews((first, second), 0.625)
    assert state.input_path == bound[0].path
    destination = tmp_path / "ektar-batch"
    expected: dict[str, tuple[bytes, bytes]] = {}
    for index, source in enumerate(bound, 1):
        stem = desktop_module._safe_output_stem(source.path)
        output = destination / f"{index:04d}-{stem}-ektar_100.png"
        destination.mkdir(exist_ok=True)
        expected[output.name] = _direct_pair(
            workflow, source.path, "ektar_100", 0.625, output
        )
    destination.rmdir()

    progress: list[tuple[int, int, str]] = []
    receipt = workflow.export_batch(
        bound,
        "ektar_100",
        destination,
        progress=lambda done, total, name: progress.append((done, total, name)),
    )
    assert progress == [(1, 2, bound[0].basename), (2, 2, bound[1].basename)]
    assert receipt.job_count == 2
    assert receipt.style_id == "ektar_100"
    assert receipt.receipt["claim"] == {
        "output_label": "film-inspired / Look Approximation",
        "evidence_grade": "look-approximation",
        "calibrated_stock_response": False,
        "physical_film_reproduction": False,
        "stock_distinguishability": False,
    }
    assert {path.suffix for path in destination.iterdir()} >= {".png", ".json"}
    assert len(list(destination.glob("*.png"))) == 2
    assert len(list(destination.glob("*.recipe.json"))) == 2
    assert {path.name for path in destination.iterdir()} == {
        "batch.json",
        *(name for name in expected),
        *(name.replace(".png", ".recipe.json") for name in expected),
    }
    receipt_text = receipt.receipt_path.read_text("utf-8")
    assert str(tmp_path.resolve()) not in receipt_text

    for row in receipt.receipt["jobs"]:
        assert Path(row["output_path"]).is_absolute() is False
        assert Path(row["recipe_path"]).is_absolute() is False
        output = destination / row["output_path"]
        recipe_path = destination / row["recipe_path"]
        assert (output.read_bytes(), recipe_path.read_bytes()) == expected[output.name]
        recipe = json.loads(recipe_path.read_text("utf-8"))
        verify_render_recipe_files(
            recipe,
            profile_path=ROOT / "configs/render_profiles/safe_rich_product_v1.json",
            root=ROOT,
        )
        replay = tmp_path / f"replay-{output.name}"
        assert replay_style_safe_recipe_to_file(
            recipe,
            output_path=replay,
            root=ROOT,
            profile_path=ROOT / "configs/render_profiles/safe_rich_product_v1.json",
        ) == sha256_file(output)
        assert replay.read_bytes() == output.read_bytes()
    assert workflow.close()


@pytest.mark.skipif(os.name != "nt", reason="formal directory publication is Windows-only")
def test_forward_reverse_selection_has_one_canonical_identity(tmp_path: Path) -> None:
    first, second = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    snapshots: list[tuple[dict[str, object], dict[str, bytes]]] = []
    destination = tmp_path / "stable-batch"
    for selected in ((first, second), (second, first)):
        _, bound = workflow.render_batch_previews(selected, 0.4)
        receipt = workflow.export_batch(bound, "portra_400", destination)
        snapshots.append(
            (
                receipt.receipt,
                {
                    path.name: path.read_bytes()
                    for path in sorted(destination.iterdir())
                },
            )
        )
        shutil.rmtree(destination)
    assert snapshots[0] == snapshots[1]
    assert workflow.close()


@pytest.mark.skipif(os.name != "nt", reason="formal directory publication is Windows-only")
def test_source_drift_rejects_before_batch_stage(tmp_path: Path) -> None:
    first, second = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews((first, second), 0.5)
    second.write_bytes(second.read_bytes() + b"drift")
    destination = tmp_path / "drift-batch"
    with pytest.raises(ProductDesktopError, match="changed after preview"):
        workflow.export_batch(bound, "velvia_50", destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".drift-batch.u7-11a-*.stage"))
    assert workflow.close()


@pytest.mark.skipif(os.name != "nt", reason="formal directory publication is Windows-only")
def test_child_nonzero_cleans_completed_children_and_stage(tmp_path: Path) -> None:
    first, second = _inputs(tmp_path)
    calls = 0

    def fail_second(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            return subprocess.CompletedProcess(command, 19, "", "injected child failure")
        return desktop_module._run_command(command, cwd, environment)

    workflow = _workflow(tmp_path, command_runner=fail_second)
    _, bound = workflow.render_batch_previews((first, second), 0.5)
    destination = tmp_path / "failed-child-batch"
    with pytest.raises(ProductDesktopError, match="child 2 failed"):
        workflow.export_batch(bound, "velvia_50", destination)
    assert calls == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-child-batch.u7-11a-*.stage"))
    assert workflow.close()


@pytest.mark.skipif(os.name != "nt", reason="formal directory publication is Windows-only")
def test_child_partial_output_is_attributed_and_cleaned(tmp_path: Path) -> None:
    first, second = _inputs(tmp_path)

    def partial_failure(command, _cwd, _environment):  # type: ignore[no-untyped-def]
        output = Path(command[-1])
        output.write_bytes(b"attributable-partial")
        return subprocess.CompletedProcess(command, 23, "", "injected partial failure")

    workflow = _workflow(tmp_path, command_runner=partial_failure)
    _, bound = workflow.render_batch_previews((first, second), 0.5)
    destination = tmp_path / "partial-child-batch"
    with pytest.raises(ProductDesktopError, match="child 1 failed"):
        workflow.export_batch(bound, "velvia_50", destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".partial-child-batch.u7-11a-*.stage"))
    assert workflow.close()


@pytest.mark.skipif(os.name != "nt", reason="formal directory publication is Windows-only")
def test_preexisting_destination_is_preserved_before_renderer(tmp_path: Path) -> None:
    first, second = _inputs(tmp_path)
    calls = 0

    def forbidden(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess([], 1, "", "forbidden")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    _, bound = workflow.render_batch_previews((first, second), 0.5)
    destination = tmp_path / "existing-batch"
    destination.mkdir()
    foreign = destination / "foreign.bin"
    foreign.write_bytes(b"foreign")
    with pytest.raises(ProductDesktopError, match="must be absent"):
        workflow.export_batch(bound, "velvia_50", destination)
    assert calls == 0
    assert foreign.read_bytes() == b"foreign"
    assert {path.name for path in destination.iterdir()} == {"foreign.bin"}
    assert workflow.close()


@pytest.mark.skipif(os.name != "nt", reason="formal directory publication is Windows-only")
def test_cancel_after_first_child_cleans_stage_without_destination(tmp_path: Path) -> None:
    first, second = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews((first, second), 0.5)
    cancel = threading.Event()

    def progress(done: int, _total: int, _name: str) -> None:
        if done == 1:
            cancel.set()

    destination = tmp_path / "cancel-batch"
    with pytest.raises(ProductDesktopError, match="cancelled safely"):
        workflow.export_batch(
            bound,
            "velvia_50",
            destination,
            cancel_event=cancel,
            progress=progress,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".cancel-batch.u7-11a-*.stage"))
    assert workflow.close()


@pytest.mark.skipif(os.name != "nt", reason="formal directory publication is Windows-only")
def test_late_foreign_destination_is_preserved(tmp_path: Path) -> None:
    first, second = _inputs(tmp_path)
    destination = tmp_path / "foreign-batch"
    calls = 0

    def runner(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal calls
        result = desktop_module._run_command(command, cwd, environment)
        calls += 1
        if calls == 2:
            destination.mkdir()
            (destination / "foreign.bin").write_bytes(b"foreign")
        return result

    workflow = _workflow(tmp_path, command_runner=runner)
    _, bound = workflow.render_batch_previews((first, second), 0.5)
    with pytest.raises(ProductDesktopError, match="destination appeared"):
        workflow.export_batch(bound, "portra_400", destination)
    assert (destination / "foreign.bin").read_bytes() == b"foreign"
    assert set(destination.iterdir()) == {destination / "foreign.bin"}
    assert not list(tmp_path.glob(".foreign-batch.u7-11a-*.stage"))
    assert workflow.close()


@pytest.mark.skipif(os.name != "nt", reason="formal directory publication is Windows-only")
def test_post_rename_verification_failure_rolls_back_owned_and_preserves_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _inputs(tmp_path)
    destination = tmp_path / "verify-failure-batch"
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews((first, second), 0.5)
    real_rename = os.rename
    real_matches = desktop_module._file_matches

    def rename_then_inject(source: Path, target: Path) -> None:
        real_rename(source, target)
        (Path(target) / "foreign.bin").write_bytes(b"foreign")

    def fail_first_final_member(seal: desktop_module._FileSeal) -> bool:
        path = seal.identity.path
        if path.parent == destination and path.name != "foreign.bin":
            return False
        return real_matches(seal)

    monkeypatch.setattr(desktop_module.os, "rename", rename_then_inject)
    monkeypatch.setattr(desktop_module, "_file_matches", fail_first_final_member)
    with pytest.raises(ProductDesktopError, match="published batch member"):
        workflow.export_batch(bound, "ektar_100", destination)
    assert destination.is_dir()
    assert {path.name for path in destination.iterdir()} == {"foreign.bin"}
    assert (destination / "foreign.bin").read_bytes() == b"foreign"
    assert not list(tmp_path.glob(".verify-failure-batch.u7-11a-*.stage"))
    assert workflow.close()


def _tk_exists(root: Any) -> bool:
    try:
        return bool(root.winfo_exists())
    except TclError:
        return False


def _pump_tk(
    root: Any, predicate: Callable[[], bool], timeout: float = 5.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        root.update()
        time.sleep(0.01)
    assert predicate()


def test_native_batch_ui_tracks_progress_cancel_and_non_daemon_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tkinter as tk

    first, second = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    root = tk.Tk()
    root.withdraw()
    shown: list[tuple[str, str]] = []
    started = threading.Event()
    release = threading.Event()
    destination = tmp_path / "ui-batch"
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showerror",
        lambda title, message: shown.append((title, message)),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showinfo",
        lambda title, message: shown.append((title, message)),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.filedialog.asksaveasfilename",
        lambda **_kwargs: str(destination),
    )
    app = build_product_desktop_app(root, workflow, initial_input=first)

    def held_export(
        inputs: object,
        style_id: str,
        output_directory: Path,
        *,
        cancel_event: threading.Event,
        progress: object,
    ) -> DesktopBatchReceipt:
        rows = tuple(inputs)  # type: ignore[arg-type]
        progress(1, len(rows), rows[0].basename)  # type: ignore[operator]
        started.set()
        assert release.wait(5)
        if cancel_event.is_set():
            raise ProductDesktopError("batch cancelled safely")
        return DesktopBatchReceipt(
            batch_id="0" * 64,
            style_id=style_id,
            look_amount=0.5,
            output_directory=output_directory,
            receipt_path=output_directory / "batch.json",
            receipt_sha256="1" * 64,
            job_count=len(rows),
            receipt={},
        )

    monkeypatch.setattr(workflow, "export_batch", held_export)
    try:
        app._set_inputs((first, second))
        state, bound = workflow.render_batch_previews((first, second), 0.5)
        app._batch_preview_complete((state, bound))
        assert app.input_text.get() == f"2 photos · previewing {bound[0].basename}"
        assert app.export_button.cget("text") == "Export 2 PNG16 + recipes"
        app.look_buttons["portra_400"].invoke()
        app.export()
        assert started.wait(2)
        worker = app._batch_thread
        assert worker is not None and worker.daemon is False
        assert app.batch_active is True and app.busy is True
        assert str(app.choose_button.cget("state")) == "disabled"
        assert str(app.preview_button.cget("state")) == "disabled"
        assert str(app.amount_scale.cget("state")) == "disabled"
        assert str(app.export_button.cget("state")) == "disabled"
        assert all(
            str(button.cget("state")) == "disabled"
            for button in app.look_buttons.values()
        )
        assert str(app.cancel_button.cget("state")) == "normal"
        _pump_tk(root, lambda: app.status.get().startswith("Rendered 1/2:"))
        app.cancel_batch()
        assert app._batch_cancel.is_set()
        assert str(app.cancel_button.cget("state")) == "disabled"
        release.set()
        _pump_tk(root, lambda: app._batch_thread is None)
        assert app.batch_active is False and app.busy is False
        assert shown[-1] == (
            "K-MCFM stopped safely",
            "batch cancelled safely",
        )
    finally:
        release.set()
        if root.winfo_exists():
            app.close()


def test_native_close_waits_for_active_batch_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tkinter as tk

    first, second = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    root = tk.Tk()
    root.withdraw()
    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.filedialog.asksaveasfilename",
        lambda **_kwargs: str(tmp_path / "close-batch"),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showerror",
        lambda *_args: None,
    )
    app = build_product_desktop_app(root, workflow, initial_input=first)

    def held_cancelled_export(
        inputs: object,
        style_id: str,
        output_directory: Path,
        *,
        cancel_event: threading.Event,
        progress: object,
    ) -> DesktopBatchReceipt:
        del inputs, style_id, output_directory, progress
        started.set()
        assert release.wait(5)
        assert cancel_event.is_set()
        raise ProductDesktopError("batch cancelled safely")

    monkeypatch.setattr(workflow, "export_batch", held_cancelled_export)
    try:
        app._set_inputs((first, second))
        state, bound = workflow.render_batch_previews((first, second), 0.5)
        app._batch_preview_complete((state, bound))
        app.look_buttons["ektar_100"].invoke()
        app.export()
        assert started.wait(2)
        app.close()
        assert app._closing is True
        assert app._batch_cancel.is_set()
        assert root.winfo_exists()
        release.set()
        _pump_tk(root, lambda: not _tk_exists(root))
        assert app._batch_thread is None
    finally:
        release.set()
        if _tk_exists(root):
            root.destroy()


def test_native_batch_success_and_error_clear_worker_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tkinter as tk

    first, _ = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    root = tk.Tk()
    root.withdraw()
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showinfo",
        lambda title, message: shown.append((title, message)),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showerror",
        lambda title, message: shown.append((title, message)),
    )
    app = build_product_desktop_app(root, workflow, initial_input=first)
    receipt = DesktopBatchReceipt(
        batch_id="0" * 64,
        style_id="portra_400",
        look_amount=0.5,
        output_directory=tmp_path / "bounded-name",
        receipt_path=tmp_path / "bounded-name" / "batch.json",
        receipt_sha256="1" * 64,
        job_count=2,
        receipt={},
    )
    try:
        app.batch_active = True
        app._set_busy(True, "Rendering 0/2 photos…")
        app._background_batch(lambda: receipt, app._batch_complete)
        _pump_tk(root, lambda: app._batch_thread is None)
        assert app.batch_active is False and app.busy is False
        assert shown[-1][0] == "Batch export complete"
        assert "film-inspired / Look Approximation" in shown[-1][1]
        assert str(tmp_path.resolve()) not in shown[-1][1]
        assert app.status.get() == "Batch complete: 2 photos in bounded-name"

        app.batch_active = True
        app._set_busy(True, "Rendering 0/2 photos…")

        def fail() -> DesktopBatchReceipt:
            raise ProductDesktopError("injected batch worker failure")

        app._background_batch(fail, app._batch_complete)
        _pump_tk(root, lambda: app._batch_thread is None)
        assert app.batch_active is False and app.busy is False
        assert shown[-1] == (
            "K-MCFM stopped safely",
            "injected batch worker failure",
        )
    finally:
        if _tk_exists(root):
            app.close()


def test_one_photo_ui_keeps_single_export_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tkinter as tk

    first, _ = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    root = tk.Tk()
    root.withdraw()
    destination = tmp_path / "single.png"
    calls: list[tuple[str, Path]] = []
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.filedialog.asksaveasfilename",
        lambda **_kwargs: str(destination),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showinfo",
        lambda title, message: shown.append((title, message)),
    )
    app = build_product_desktop_app(root, workflow, initial_input=first)
    state = workflow.render_previews(first, 0.5)
    app._preview_complete(state)

    def single_export(style_id: str, output_path: Path) -> object:
        calls.append((style_id, output_path))
        return desktop_module.DesktopExportReceipt(
            style_id=style_id,
            look_amount=0.5,
            input_path=first.resolve(),
            input_sha256=sha256_file(first),
            output_path=output_path,
            output_sha256="1" * 64,
            recipe_path=output_path.with_suffix(".recipe.json"),
            recipe_sha256="2" * 64,
            recipe={},
        )

    monkeypatch.setattr(workflow, "export", single_export)
    monkeypatch.setattr(
        workflow,
        "export_batch",
        lambda *_args, **_kwargs: pytest.fail("single photo routed to batch"),
    )
    monkeypatch.setattr(app, "_background", lambda action, success: success(action()))
    try:
        app.look_buttons["velvia_50"].invoke()
        app.export()
        assert calls == [("velvia_50", destination)]
        assert shown[-1][0] == "Export complete"
        assert app._batch_thread is None and app.batch_active is False
    finally:
        if _tk_exists(root):
            app.close()
