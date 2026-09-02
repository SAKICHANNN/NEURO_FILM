from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from src.inference.product_desktop import ProductDesktopError
from src.inference.product_desktop_ui import build_product_desktop_app


def _preview_png(colour: tuple[int, int, int]) -> bytes:
    encoded = BytesIO()
    Image.new("RGB", (8, 6), colour).save(encoded, format="PNG")
    return encoded.getvalue()


class _PreviewWorkflow:
    def __init__(self) -> None:
        self.preview_state: Any | None = None
        self._previews = {
            "velvia_50": _preview_png((210, 40, 30)),
            "portra_400": _preview_png((180, 120, 100)),
            "ektar_100": _preview_png((40, 90, 210)),
        }

    def bind_batch_inputs(self, paths: object) -> tuple[Any, ...]:
        return tuple(
            SimpleNamespace(path=Path(path).resolve(), basename=Path(path).name)
            for path in paths  # type: ignore[union-attr]
        )

    def input_preview_bytes(self) -> bytes:
        return _preview_png((90, 100, 110))

    def preview_bytes(self) -> dict[str, bytes]:
        return dict(self._previews)

    def close(self) -> bool:
        self.preview_state = None
        return True


def test_error_clears_every_invalidated_preview_widget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tkinter as tk

    source = tmp_path / "source.png"
    source.write_bytes(_preview_png((12, 34, 56)))
    workflow = _PreviewWorkflow()
    root = tk.Tk()
    root.withdraw()
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showerror",
        lambda title, message: shown.append((str(title), str(message))),
    )
    app = build_product_desktop_app(root, workflow)  # type: ignore[arg-type]
    try:
        app._set_input(source)
        state = SimpleNamespace(input_path=source.resolve())
        workflow.preview_state = state
        app._preview_complete(state)  # type: ignore[arg-type]
        app.look_buttons["portra_400"].invoke()

        assert app.preview_ready is True
        assert app.input_preview_image is not None
        assert len(app.preview_images) == 3
        assert all(not str(label.cget("text")) for label in app.preview_labels.values())

        app._show_error(ProductDesktopError("injected safe stop"))

        assert shown == [("K-MCFM stopped safely", "injected safe stop")]
        assert app.status.get() == "Stopped safely: injected safe stop"
        assert app.preview_ready is False
        assert app.style.get() == ""
        assert app.input_preview_image is None
        assert app.preview_images == []
        assert all(
            label.cget("text") == "Preview not rendered"
            and not str(label.cget("image"))
            for label in app.preview_labels.values()
        )
        assert all(
            str(button.cget("state")) == "disabled"
            for button in app.look_buttons.values()
        )
        assert str(app.export_button.cget("state")) == "disabled"
        assert str(app.detail_button.cget("state")) == "disabled"
        # The owned workspace remains available for the next explicit preview
        # render; only its no-longer-authoritative visible cards are removed.
        assert workflow.preview_state is state
    finally:
        if root.winfo_exists():
            app.close()
