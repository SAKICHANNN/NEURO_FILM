from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.inference.product_desktop import (
    PRODUCT_OUTPUT_FORMATS,
    DesktopExportReceipt,
    ProductDesktopError,
    ProductDesktopWorkflow,
)
from src.inference.product_desktop_ui import build_product_desktop_app
from src.inference.render_contract import sha256_file
from src.inference.style_safe_engine import replay_style_safe_recipe_to_file

ROOT = Path(__file__).resolve().parents[1]
_SHARED_TK_ROOT: Any | None = None


def _shared_tk_root() -> Any:
    import tkinter as tk

    global _SHARED_TK_ROOT
    if _SHARED_TK_ROOT is None:
        _SHARED_TK_ROOT = tk.Tk()
        _SHARED_TK_ROOT.withdraw()
        # Keep the one Tcl/Tk interpreter alive for the pytest process.  The
        # app under test still owns and destroys only its child Toplevels.
    return _SHARED_TK_ROOT


def _source(path: Path, offset: int = 0) -> None:
    yy, xx = np.mgrid[:37, :53]
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
        max_preview_pixels=2_000,
        tile_size=23,
        tile_workers=1,
        png_compression=6,
        **kwargs,  # type: ignore[arg-type]
    )


def _environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
    }


def _pump(root: Any, predicate: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        root.update()
        time.sleep(0.01)
    raise AssertionError("Tk condition did not become true")


def test_contract_and_dependency_lock_are_exact() -> None:
    contract = json.loads(
        (ROOT / "configs/u7_12b_desktop_single_photo_output_format_v1.json").read_text(
            "utf-8"
        )
    )
    assert contract["dependency_gate"]["status"] == (
        "PASS_PRIVATE_U7_12A_DESKTOP_FOREGROUND_CLOSE_SAFETY"
    )
    assert contract["dependency_gate"]["artifact_sha256"]["evidence"] == sha256_file(
        ROOT / "docs/evidence/U7_12A_DESKTOP_FOREGROUND_WORKER_CLOSE_SAFETY_RESULT.json"
    )
    assert tuple(row.format_id for row in PRODUCT_OUTPUT_FORMATS) == (
        "png16",
        "tiff16",
        "jpeg8",
    )
    assert contract["claim_ceiling"]["calibrated_stock_response"] is False
    assert contract["claim_ceiling"]["physical_film_reproduction"] is False


@pytest.mark.parametrize(
    ("format_id", "suffix", "recipe_format", "bit_depth"),
    [
        ("png16", ".png", "PNG", 16),
        ("tiff16", ".tiff", "TIFF", 16),
        ("jpeg8", ".jpg", "JPEG", 8),
    ],
)
def test_single_export_matches_direct_cli_and_strict_replay(
    tmp_path: Path,
    format_id: str,
    suffix: str,
    recipe_format: str,
    bit_depth: int,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    workflow = _workflow(tmp_path)
    workflow.render_previews(source, 0.4)
    output = tmp_path / f"exact{suffix}"
    command = workflow.export_command(
        "ektar_100",
        output,
        output_format_id=format_id,
    )
    direct = subprocess.run(
        command,
        cwd=ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert direct.returncode == 0, direct.stderr
    direct_image = output.read_bytes()
    direct_recipe = output.with_suffix(".recipe.json").read_bytes()
    output.unlink()
    output.with_suffix(".recipe.json").unlink()

    receipt = workflow.export(
        "ektar_100",
        output,
        output_format_id=format_id,
    )
    assert receipt.output_format_id == format_id
    assert output.read_bytes() == direct_image
    assert receipt.recipe_path.read_bytes() == direct_recipe
    assert receipt.recipe["output"]["format"] == recipe_format
    assert receipt.recipe["output"]["bit_depth"] == bit_depth
    assert Path(receipt.recipe["output"]["path"]).resolve() == output.resolve()
    assert receipt.recipe["claim"]["evidence_grade"] == "look-approximation"
    assert receipt.recipe["claim"]["calibrated_reference_allowed"] is False

    replay = tmp_path / f"replay{suffix}"
    digest = replay_style_safe_recipe_to_file(
        receipt.recipe,
        profile_path=ROOT / "configs/render_profiles/safe_rich_product_v1.json",
        output_path=replay,
        root=ROOT,
    )
    assert replay.read_bytes() == direct_image
    assert digest == sha256_file(replay)
    assert workflow.close()
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize(
    ("format_id", "name", "message"),
    [
        ("unknown", "output.png", "unknown desktop output format"),
        ("png16", "output", "requires a file extension"),
        ("png16", "output.tiff", "PNG16 export requires"),
        ("tiff16", "output.png", "TIFF16 export requires"),
        ("jpeg8", "output.tiff", "JPEG8 export requires"),
    ],
)
def test_invalid_format_or_suffix_rejects_before_command_runner(
    tmp_path: Path,
    format_id: str,
    name: str,
    message: str,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    calls = 0

    def forbidden(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        raise AssertionError("renderer must not run")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    workflow.render_previews(source, 0.5)
    destination = tmp_path / name
    with pytest.raises(ProductDesktopError, match=message):
        workflow.export(
            "velvia_50",
            destination,
            output_format_id=format_id,
        )
    assert calls == 0
    assert not destination.exists()
    assert not destination.with_suffix(".recipe.json").exists()
    assert workflow.close()


def test_png16_default_command_and_batch_builder_remain_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    workflow = _workflow(tmp_path)
    output = tmp_path / "output.png"
    default = workflow._build_export_command(source, "portra_400", output, 0.25)
    explicit = workflow._build_export_command(
        source,
        "portra_400",
        output,
        0.25,
        output_format_id="png16",
    )
    assert default == explicit
    assert default[default.index("--output-bit-depth") + 1] == "16"
    assert default[default.index("--png-compression") + 1] == "6"
    workflow.png_compression = 4
    custom_png = workflow._build_export_command(
        source,
        "portra_400",
        output,
        0.25,
        output_format_id="png16",
    )
    assert custom_png[custom_png.index("--png-compression") + 1] == "4"
    for format_id, suffix, bit_depth in (
        ("tiff16", ".tiff", "16"),
        ("jpeg8", ".jpg", "8"),
    ):
        command = workflow._build_export_command(
            source,
            "portra_400",
            tmp_path / f"output{suffix}",
            0.25,
            output_format_id=format_id,
        )
        assert command[command.index("--output-bit-depth") + 1] == bit_depth
        assert "--png-compression" not in command


def test_desktop_format_controls_route_single_photo_and_fix_batch_png16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tkinter as tk

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _source(first)
    _source(second, 7)
    workflow = _workflow(tmp_path)
    root = tk.Toplevel(_shared_tk_root())
    root.withdraw()
    dialogs: list[dict[str, Any]] = []
    calls: list[tuple[str, Path, str]] = []
    shown: list[tuple[str, str]] = []

    def choose(**kwargs: Any) -> str:
        dialogs.append(kwargs)
        format_id = app.output_format.get()
        suffix = {"png16": ".png", "tiff16": ".tiff", "jpeg8": ".jpg"}[format_id]
        return str(tmp_path / f"selected-{format_id}{suffix}")

    def export(
        style_id: str,
        output_path: Path,
        *,
        output_format_id: str = "png16",
    ) -> DesktopExportReceipt:
        calls.append((style_id, output_path, output_format_id))
        return DesktopExportReceipt(
            style_id=style_id,
            look_amount=0.5,
            input_path=first.resolve(),
            input_sha256=sha256_file(first),
            output_path=output_path,
            output_sha256="1" * 64,
            recipe_path=output_path.with_suffix(".recipe.json"),
            recipe_sha256="2" * 64,
            recipe={},
            output_format_id=output_format_id,
        )

    monkeypatch.setattr(
        "src.inference.product_desktop_ui.filedialog.asksaveasfilename", choose
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showinfo",
        lambda title, message: shown.append((title, message)),
    )
    monkeypatch.setattr(workflow, "export", export)
    app = build_product_desktop_app(root, workflow)
    monkeypatch.setattr(app, "_background", lambda action, success: success(action()))
    try:
        assert app.output_format_text.get() == "Choose one photo for output format"
        assert all(
            str(button.cget("state")) == "disabled"
            for button in app.output_format_buttons.values()
        )
        app._set_input(first)
        assert app.output_format_text.get() == "Single-photo output"
        state = workflow.render_previews(first, 0.5)
        app._preview_complete(state)
        app.look_buttons["ektar_100"].invoke()
        original_state = workflow.preview_state
        for format_id in ("png16", "tiff16", "jpeg8"):
            app.output_format_buttons[format_id].invoke()
            assert workflow.preview_state is original_state
            assert app.preview_ready is True
            assert str(app.output_format_buttons[format_id].cget("state")) == "normal"
            assert bool(app.output_format_buttons[format_id].cget("takefocus"))
            app.export()
            assert calls[-1][2] == format_id
            assert format_id.removesuffix("16").removesuffix("8").upper() in str(
                app.export_button.cget("text")
            )

        assert [row[2] for row in calls] == ["png16", "tiff16", "jpeg8"]
        assert [row["defaultextension"] for row in dialogs] == [
            ".png",
            ".tiff",
            ".jpg",
        ]
        assert shown[-1][0] == "Export complete"

        app._set_inputs((first, second))
        assert app.output_format.get() == "png16"
        assert app.output_format_text.get() == "Batch output fixed: PNG16"
        assert all(
            str(button.cget("state")) == "disabled"
            for button in app.output_format_buttons.values()
        )
        assert "2 PNG16" in str(app.export_button.cget("text"))
    finally:
        if root.winfo_exists():
            app.close()


class _HeldWorkflow:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release
        self.preview_state: Any = None
        self.close_count = 0
        self.received_format = ""

    def export(
        self,
        _style_id: str,
        _destination: Path,
        *,
        output_format_id: str = "png16",
    ) -> object:
        self.received_format = output_format_id
        self.started.set()
        assert self.release.wait(5)
        return object()

    def close(self) -> bool:
        self.close_count += 1
        return True


@pytest.mark.parametrize(
    ("format_id", "suffix"),
    [("png16", ".png"), ("tiff16", ".tiff"), ("jpeg8", ".jpg")],
)
def test_close_waits_safely_for_each_single_photo_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_id: str,
    suffix: str,
) -> None:
    import tkinter as tk

    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    started = threading.Event()
    release = threading.Event()
    workflow = _HeldWorkflow(started, release)
    workflow.preview_state = SimpleNamespace(input_path=source)
    root = tk.Toplevel(_shared_tk_root())
    root.withdraw()
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.filedialog.asksaveasfilename",
        lambda **_kwargs: str(tmp_path / f"output{suffix}"),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showerror",
        lambda *_args: pytest.fail("close must not show an error"),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showinfo",
        lambda *_args: pytest.fail("close must suppress success"),
    )
    app = build_product_desktop_app(root, workflow)  # type: ignore[arg-type]
    try:
        app.input_path = source
        app.input_paths = (source,)
        app.batch_inputs = (object(),)  # type: ignore[assignment]
        app.preview_ready = True
        app.style.set("ektar_100")
        app.output_format.set(format_id)
        app._update_output_format_controls()
        app.export()
        assert started.wait(2)
        app.close()
        assert workflow.close_count == 0
        assert "Closing safely after the current operation" in app.status.get()
        release.set()
        _pump(root, lambda: workflow.close_count == 1)
        assert workflow.received_format == format_id
    finally:
        release.set()
        if workflow.close_count == 0 and root.winfo_exists():
            app.close()
