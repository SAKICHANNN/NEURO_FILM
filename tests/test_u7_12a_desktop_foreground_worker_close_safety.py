from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.inference.product_desktop import ProductDesktopError
from src.inference.product_desktop_ui import build_product_desktop_app

_SHARED_TK_ROOT: Any | None = None


@pytest.fixture(scope="module", autouse=True)
def _close_shared_tk_root_after_module() -> Any:
    yield
    if _SHARED_TK_ROOT is not None:
        _SHARED_TK_ROOT.destroy()


def _shared_tk_root() -> Any:
    import tkinter as tk

    global _SHARED_TK_ROOT
    if _SHARED_TK_ROOT is None:
        _SHARED_TK_ROOT = tk.Tk()
        _SHARED_TK_ROOT.withdraw()
    return _SHARED_TK_ROOT


class _HeldWorkflow:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release
        self.preview_state: Any | None = None
        self.close_count = 0

    def render_batch_previews(self, _sources: object, _amount: float) -> object:
        self.started.set()
        assert self.release.wait(5)
        return object()

    def export(self, _style_id: str, _destination: Path) -> object:
        self.started.set()
        assert self.release.wait(5)
        return object()

    def close(self) -> bool:
        self.close_count += 1
        return True


def _pump(root: Any, predicate: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        try:
            root.update()
        except Exception:
            if predicate():
                return
            raise
        time.sleep(0.01)
    raise AssertionError("Tk condition did not become true")


@pytest.mark.parametrize("operation", ["preview", "single-export"])
def test_close_waits_asynchronously_for_foreground_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    import tkinter as tk

    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    started = threading.Event()
    release = threading.Event()
    workflow = _HeldWorkflow(started, release)
    root = tk.Toplevel(_shared_tk_root())
    root.withdraw()
    errors: list[tuple[str, str]] = []
    infos: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showerror",
        lambda title, message: errors.append((title, message)),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showinfo",
        lambda title, message: infos.append((title, message)),
    )
    app = build_product_desktop_app(root, workflow)  # type: ignore[arg-type]
    try:
        if operation == "preview":
            app._set_inputs((source,))
            app.render_previews()
        else:
            destination = tmp_path / "output.png"
            monkeypatch.setattr(
                "src.inference.product_desktop_ui.filedialog.asksaveasfilename",
                lambda **_kwargs: str(destination),
            )
            workflow.preview_state = SimpleNamespace(input_path=source)
            app.input_path = source
            app.input_paths = (source,)
            app.batch_inputs = (object(),)  # type: ignore[assignment]
            app.preview_ready = True
            app.style.set("ektar_100")
            app.export()

        assert started.wait(2)
        worker = app._foreground_thread
        assert worker is not None
        assert worker.daemon is False
        before = time.monotonic()
        app.close()
        assert time.monotonic() - before < 0.2
        assert app._closing is True
        assert workflow.close_count == 0
        assert "Closing safely after the current operation" in app.status.get()

        release.set()
        _pump(root, lambda: workflow.close_count == 1)
        assert app._foreground_thread is None
        assert errors == []
        assert infos == []
    finally:
        release.set()
        if workflow.close_count == 0:
            if app._foreground_thread is not None:
                _pump(root, lambda: app._foreground_thread is None)
            if root.winfo_exists():
                app.close()


def test_foreground_success_error_and_concurrency_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tkinter as tk

    started = threading.Event()
    release = threading.Event()
    workflow = _HeldWorkflow(started, release)
    root = tk.Toplevel(_shared_tk_root())
    root.withdraw()
    shown: list[str] = []
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showerror",
        lambda _title, message: shown.append(str(message)),
    )
    app = build_product_desktop_app(root, workflow)  # type: ignore[arg-type]
    successes: list[str] = []
    try:
        app._background(
            lambda: (started.set(), release.wait(5), "first")[-1],
            successes.append,
        )
        assert started.wait(2)
        first = app._foreground_thread
        assert first is not None and first.daemon is False
        with pytest.raises(ProductDesktopError, match="already active"):
            app._background(lambda: "second", successes.append)
        assert app._foreground_thread is first
        release.set()
        _pump(root, lambda: app._foreground_thread is None)
        assert successes == ["first"]

        def fail() -> None:
            raise ProductDesktopError("foreground failure")

        app._background(fail, lambda _result: successes.append("unexpected"))
        _pump(root, lambda: app._foreground_thread is None)
        assert shown == ["foreground failure"]
        assert successes == ["first"]
    finally:
        release.set()
        if root.winfo_exists():
            app.close()
