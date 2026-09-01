from __future__ import annotations

import threading
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

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


def _preview_png(red: int, green: int, blue: int) -> bytes:
    encoded = BytesIO()
    Image.new("RGB", (4, 3), (red, green, blue)).save(encoded, format="PNG")
    return encoded.getvalue()


class _ErrorWorkflow:
    def __init__(self) -> None:
        self.preview_state: Any | None = None
        self.closed = 0
        self._previews = {
            "velvia_50": _preview_png(210, 40, 30),
            "portra_400": _preview_png(180, 120, 100),
            "ektar_100": _preview_png(40, 90, 210),
        }

    def bind_batch_inputs(self, paths: object) -> tuple[Any, ...]:
        return tuple(
            SimpleNamespace(path=Path(path).resolve(), basename=Path(path).name)
            for path in paths  # type: ignore[union-attr]
        )

    def preview_bytes(self) -> dict[str, bytes]:
        return dict(self._previews)

    def export(
        self,
        _style_id: str,
        _destination: Path,
        *,
        output_format_id: str = "png16",
    ) -> object:
        assert output_format_id == "png16"
        raise ProductDesktopError("injected single export failure")

    def export_batch(
        self,
        _inputs: object,
        _style_id: str,
        _destination: Path,
        *,
        output_format_id: str = "png16",
        cancel_event: threading.Event,
        progress: object,
    ) -> object:
        assert output_format_id == "png16"
        del cancel_event, progress
        raise ProductDesktopError("batch cancelled safely")

    def close(self) -> bool:
        self.preview_state = None
        self.closed += 1
        return True


def _pump(root: Any, predicate: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        root.update()
        time.sleep(0.01)
    raise AssertionError("Tk condition did not become true")


@pytest.mark.parametrize("operation", ["single-export-error", "batch-error"])
def test_error_invalidates_stale_selection_before_new_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    import tkinter as tk

    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    workflow = _ErrorWorkflow()
    root = tk.Toplevel(_shared_tk_root())
    root.withdraw()
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showerror",
        lambda title, message: shown.append((str(title), str(message))),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showinfo",
        lambda *_args: pytest.fail("an error path must not show success"),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.filedialog.asksaveasfilename",
        lambda **_kwargs: str(tmp_path / "destination"),
    )
    app = build_product_desktop_app(root, workflow)  # type: ignore[arg-type]
    paths = (first,) if operation == "single-export-error" else (first, second)
    try:
        app._set_inputs(paths)
        state = SimpleNamespace(input_path=first.resolve())
        workflow.preview_state = state
        bound = workflow.bind_batch_inputs(paths)
        app._batch_preview_complete((state, bound))  # type: ignore[arg-type]
        app.look_buttons["portra_400"].invoke()
        assert app.style.get() == "portra_400"
        assert str(app.export_button.cget("state")) == "normal"

        app.export()
        if operation == "single-export-error":
            _pump(root, lambda: app._foreground_thread is None)
            expected_error = "injected single export failure"
        else:
            _pump(root, lambda: app._batch_thread is None)
            expected_error = "batch cancelled safely"

        assert shown[-1] == ("K-MCFM stopped safely", expected_error)
        assert app.preview_ready is False
        assert app.style.get() == ""
        assert str(app.export_button.cget("state")) == "disabled"

        new_state = SimpleNamespace(input_path=first.resolve())
        workflow.preview_state = new_state
        app._batch_preview_complete((new_state, bound))  # type: ignore[arg-type]
        assert app.preview_ready is True
        assert app.style.get() == ""
        assert str(app.export_button.cget("state")) == "disabled"
        assert all(
            str(button.cget("state")) == "normal"
            for button in app.look_buttons.values()
        )

        app.look_buttons["ektar_100"].invoke()
        assert app.style.get() == "ektar_100"
        assert str(app.export_button.cget("state")) == "normal"
    finally:
        if root.winfo_exists():
            app.close()


def test_success_completion_does_not_clear_valid_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tkinter as tk

    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    workflow = _ErrorWorkflow()
    root = tk.Toplevel(_shared_tk_root())
    root.withdraw()
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showinfo",
        lambda *_args: None,
    )
    app = build_product_desktop_app(root, workflow)  # type: ignore[arg-type]
    try:
        app._set_inputs((source,))
        state = SimpleNamespace(input_path=source.resolve())
        workflow.preview_state = state
        app._preview_complete(state)  # type: ignore[arg-type]
        app.look_buttons["velvia_50"].invoke()
        app._export_complete(
            SimpleNamespace(
                output_path=tmp_path / "output.png",
                recipe_path=tmp_path / "output.recipe.json",
            )
        )
        assert app.preview_ready is True
        assert app.style.get() == "velvia_50"
        assert str(app.export_button.cget("state")) == "normal"
    finally:
        if root.winfo_exists():
            app.close()
