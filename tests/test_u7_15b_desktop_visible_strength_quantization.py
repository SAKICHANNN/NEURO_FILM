from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.inference.product_desktop import ProductDesktopError, ProductDesktopWorkflow
from src.inference.product_desktop_ui import (
    _visible_look_amount,
    build_product_desktop_app,
)

ROOT = Path(__file__).resolve().parents[1]
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


@pytest.mark.parametrize(
    ("value", "expected_percent", "expected_amount"),
    (
        (0.654321, 65, 0.65),
        (0.625, 63, 0.63),
        (0.624999, 62, 0.62),
        (0.005, 1, 0.01),
        (0.004999, 0, 0.0),
        (-0.25, 0, 0.0),
        (1.25, 100, 1.0),
    ),
)
def test_visible_amount_uses_frozen_half_up_percent(
    value: float, expected_percent: int, expected_amount: float
) -> None:
    assert _visible_look_amount(value) == (expected_percent, expected_amount)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_visible_amount_rejects_non_finite(value: float) -> None:
    with pytest.raises(ProductDesktopError, match="look strength must be finite"):
        _visible_look_amount(value)


class _WorkflowProbe:
    def __init__(self) -> None:
        self.preview_state: Any | None = None
        self.calls: list[tuple[tuple[Path, ...], float, Path | None]] = []

    def render_batch_previews(
        self,
        sources: tuple[Path, ...],
        amount: float,
        *,
        representative_path: Path | None = None,
    ) -> object:
        self.calls.append((sources, amount, representative_path))
        return object()


def _app(tmp_path: Path) -> tuple[Any, Any, _WorkflowProbe]:
    import tkinter as tk

    source = tmp_path / "source.png"
    source.write_bytes(b"source identity only")
    workflow = _WorkflowProbe()
    root = tk.Toplevel(_shared_tk_root())
    root.withdraw()
    app = build_product_desktop_app(root, workflow)  # type: ignore[arg-type]
    app._set_input(source)
    return root, app, workflow


def test_callback_and_render_entry_share_one_visible_amount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, app, workflow = _app(tmp_path)
    monkeypatch.setattr(app, "_background", lambda action, _success: action())
    try:
        app.amount.set(0.625)
        app._amount_changed()
        assert app.amount.get() == 0.63
        assert app.amount_label.cget("text") == "63%"

        app.amount.set(0.654321)
        app.render_previews()
        assert app.amount.get() == 0.65
        assert app.amount_label.cget("text") == "65%"
        assert workflow.calls == [((app.input_path,), 0.65, None)]
    finally:
        if root.winfo_exists():
            root.destroy()


def test_non_finite_render_rejects_before_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, app, workflow = _app(tmp_path)
    monkeypatch.setattr(app, "_background", lambda action, _success: action())
    try:
        app.amount.set(float("nan"))
        with pytest.raises(ProductDesktopError, match="look strength must be finite"):
            app.render_previews()
        assert workflow.calls == []
        assert app.busy is False
    finally:
        if root.winfo_exists():
            root.destroy()


def test_real_preview_export_and_recipe_use_the_visible_amount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tkinter as tk

    source = tmp_path / "real-source.png"
    yy, xx = np.mgrid[:29, :41]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3) % 251,
            (xx * 5 + yy * 13 + 17) % 251,
            (xx * 7 + yy * 19 + 41) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(source)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    workflow = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=4_000,
        tile_size=23,
    )
    root = tk.Toplevel(_shared_tk_root())
    root.withdraw()
    app = build_product_desktop_app(root, workflow, initial_input=source)
    monkeypatch.setattr(app, "_background", lambda action, success: success(action()))
    try:
        app.amount.set(0.654321)
        app.render_previews()
        assert app.amount.get() == 0.65
        assert app.amount_label.cget("text") == "65%"
        assert workflow.preview_state is not None
        assert workflow.preview_state.look_amount == 0.65

        output = tmp_path / "ektar-visible-strength.png"
        receipt = workflow.export("ektar_100", output)
        recipe = json.loads(receipt.recipe_path.read_text("utf-8"))
        assert receipt.look_amount == 0.65
        assert recipe["render"]["look_amount"] == 0.65
        assert output.is_file()
        assert receipt.recipe_path.is_file()
    finally:
        if root.winfo_exists():
            app.close()
    assert list(scratch.iterdir()) == []
