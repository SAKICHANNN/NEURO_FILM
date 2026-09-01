from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.product_desktop import (
    PRODUCT_LOOKS,
    DesktopExportReceipt,
    ProductDesktopError,
    ProductDesktopWorkflow,
)
from src.inference.product_desktop_ui import build_product_desktop_app
from src.inference.render_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path) -> None:
    yy, xx = np.mgrid[:57, :83]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3) % 251,
            (xx * 5 + yy * 13 + 17) % 251,
            (xx * 7 + yy * 19 + 41) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _workflow(tmp_path: Path, **kwargs: object) -> ProductDesktopWorkflow:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=4_000,
        tile_size=23,
        **kwargs,  # type: ignore[arg-type]
    )


def test_product_desktop_catalog_is_explicit_and_claim_bounded() -> None:
    assert [row["style_id"] for row in PRODUCT_LOOKS] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
    ]
    assert all(
        row["claim"] == "film-inspired / Look Approximation" for row in PRODUCT_LOOKS
    )
    config = json.loads(
        (ROOT / "configs/u7_10a_product_desktop_input_workflow_v1.json").read_text(
            "utf-8"
        )
    )
    assert config["claim_ceiling"]["calibrated_stock_response"] is False
    assert config["claim_ceiling"]["physical_film_reproduction"] is False


@pytest.mark.parametrize("amount", [-0.01, 1.01, math.inf, math.nan, True, "0.5"])
def test_invalid_amount_rejects_before_preview(amount: object, tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    called = False

    def forbidden(*args: object, **kwargs: object) -> dict:
        nonlocal called
        called = True
        return {}

    workflow = _workflow(tmp_path, preview_renderer=forbidden)
    with pytest.raises(ProductDesktopError, match="finite number"):
        workflow.render_previews(source, amount)  # type: ignore[arg-type]
    assert called is False


def test_three_previews_are_distinct_bound_and_cleaned(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    workflow = _workflow(tmp_path)
    state = workflow.render_previews(source, 0.65)
    assert state.input_sha256 == sha256_file(source)
    assert state.look_amount == 0.65
    assert state.manifest["preview_pixels"] <= 4_000
    assert [row["style_id"] for row in state.manifest["rows"]] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
    ]
    assert len({row["output_sha256"] for row in state.manifest["rows"]}) == 3
    assert workflow.close() is True
    assert list((tmp_path / "scratch").iterdir()) == []


def test_export_requires_current_preview_and_preserves_existing_pair(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    calls = 0

    def forbidden(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1, "", "forbidden")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    with pytest.raises(ProductDesktopError, match="render previews"):
        workflow.export("velvia_50", tmp_path / "out.png")
    workflow.render_previews(source, 1.0)
    output = tmp_path / "out.png"
    recipe = tmp_path / "out.recipe.json"
    output.write_bytes(b"foreign-image")
    recipe.write_bytes(b"foreign-recipe")
    with pytest.raises(ProductDesktopError, match="must be absent"):
        workflow.export("velvia_50", output)
    assert output.read_bytes() == b"foreign-image"
    assert recipe.read_bytes() == b"foreign-recipe"
    assert calls == 0
    assert workflow.close()


@pytest.mark.parametrize("relative", ["inside.png", "previews/inside.png"])
def test_export_rejects_current_preview_workspace_before_renderer(
    tmp_path: Path, relative: str
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    calls = 0

    def forbidden(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1, "", "forbidden")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    state = workflow.render_previews(source, 0.5)
    destination = state.workspace / relative
    with pytest.raises(ProductDesktopError, match="outside the preview workspace"):
        workflow.export("portra_400", destination)
    assert calls == 0
    assert not destination.exists()
    assert not destination.with_suffix(".recipe.json").exists()
    assert workflow.close() is True
    assert list((tmp_path / "scratch").iterdir()) == []


def test_input_drift_and_unknown_look_fail_before_renderer(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    calls = 0

    def forbidden(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1, "", "forbidden")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    workflow.render_previews(source, 0.5)
    with pytest.raises(ProductDesktopError, match="unknown"):
        workflow.export("unknown", tmp_path / "unknown.png")
    source.write_bytes(source.read_bytes() + b"drift")
    with pytest.raises(ProductDesktopError, match="changed after preview"):
        workflow.export("portra_400", tmp_path / "drift.png")
    assert calls == 0
    assert workflow.close() is True


def test_replaced_preview_member_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    workflow = _workflow(tmp_path)
    state = workflow.render_previews(source, 0.75)
    target = Path(state.manifest["rows"][0]["output_path"])
    target.unlink()
    target.write_bytes(b"foreign")
    assert workflow.close() is False
    assert target.read_bytes() == b"foreign"


def test_replaced_preview_blocks_display_and_export_before_renderer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    calls = 0

    def forbidden(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1, "", "forbidden")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    state = workflow.render_previews(source, 0.75)
    target = Path(state.manifest["rows"][0]["output_path"])
    target.unlink()
    target.write_bytes(b"foreign")
    with pytest.raises(ProductDesktopError, match="ownership changed"):
        workflow.preview_bytes()
    with pytest.raises(ProductDesktopError, match="ownership changed"):
        workflow.export("velvia_50", tmp_path / "blocked.png")
    assert calls == 0
    assert target.read_bytes() == b"foreign"


def test_renderer_failure_preserves_unbound_foreign_member(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    foreign: Path | None = None

    def failing_renderer(
        _source_path: Path, output_directory: Path, **_kwargs: object
    ) -> dict:
        nonlocal foreign
        output_directory.mkdir()
        foreign = output_directory / "foreign.bin"
        foreign.write_bytes(b"foreign")
        raise RuntimeError("injected failure")

    workflow = _workflow(tmp_path, preview_renderer=failing_renderer)
    with pytest.raises(RuntimeError, match="injected failure"):
        workflow.render_previews(source, 0.5)
    assert foreign is not None
    assert foreign.read_bytes() == b"foreign"


def test_session_asset_drift_blocks_preview_and_export_before_renderer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    binding = tmp_path / "session-asset.json"
    binding.write_text('{"version":1}\n', encoding="utf-8")
    calls = 0

    def forbidden(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1, "", "forbidden")

    workflow = _workflow(
        tmp_path,
        command_runner=forbidden,
        session_binding_paths=(binding,),
    )
    workflow.render_previews(source, 0.5)
    binding.write_text('{"version":2}\n', encoding="utf-8")
    with pytest.raises(ProductDesktopError, match="renderer session changed"):
        workflow.preview_bytes()
    with pytest.raises(ProductDesktopError, match="renderer session changed"):
        workflow.export("ektar_100", tmp_path / "blocked.png")
    assert calls == 0
    assert workflow.close() is True


def test_real_export_matches_same_path_direct_cli_exactly(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    workflow = _workflow(tmp_path)
    workflow.render_previews(source, 0.4)
    output = tmp_path / "exact.png"
    command = workflow.export_command("ektar_100", output)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
    }
    direct = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert direct.returncode == 0, direct.stderr
    direct_image = output.read_bytes()
    direct_recipe = output.with_suffix(".recipe.json").read_bytes()
    output.unlink()
    output.with_suffix(".recipe.json").unlink()

    receipt = workflow.export("ektar_100", output)
    assert output.read_bytes() == direct_image
    assert receipt.recipe_path.read_bytes() == direct_recipe
    assert receipt.output_sha256 == sha256_file(output)
    assert receipt.recipe["claim"]["evidence_grade"] == "look-approximation"
    assert receipt.recipe["claim"]["calibrated_reference_allowed"] is False
    assert workflow.close()


def test_native_window_covers_bounded_states_without_path_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tkinter as tk

    source = tmp_path / "private-source-name.png"
    _source(source)
    workflow = _workflow(tmp_path)
    root = tk.Tk()
    root.withdraw()
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showerror",
        lambda title, message: shown.append((title, message)),
    )
    monkeypatch.setattr(
        "src.inference.product_desktop_ui.messagebox.showinfo",
        lambda title, message: shown.append((title, message)),
    )
    app = build_product_desktop_app(root, workflow, initial_input=source)
    try:
        assert root.title() == "K-MCFM Look Approximation"
        assert app.input_text.get() == source.name
        assert str(source.parent) not in app.input_text.get()
        assert tuple(app.preview_labels) == tuple(
            row["style_id"] for row in PRODUCT_LOOKS
        )
        assert str(app.export_button.cget("state")) == "disabled"

        app._set_busy(True, "Rendering")
        assert app.busy is True
        assert str(app.choose_button.cget("state")) == "disabled"
        assert str(app.preview_button.cget("state")) == "disabled"

        state = workflow.render_previews(source, 0.6)
        app._preview_complete(state)
        root.update_idletasks()
        assert app.busy is False
        assert str(app.export_button.cget("state")) == "normal"
        assert len(app.preview_images) == 3
        assert all(
            photo.width() <= 300 and photo.height() <= 260
            for photo in app.preview_images
        )
        assert all(not str(label.cget("text")) for label in app.preview_labels.values())
        assert app.choose_button.cget("style") == "Secondary.TButton"
        assert app.preview_button.cget("style") == "Secondary.TButton"
        assert app.export_button.cget("style") == "Primary.TButton"
        assert bool(app.choose_button.cget("takefocus"))
        assert bool(app.preview_button.cget("takefocus"))
        assert bool(app.export_button.cget("takefocus"))

        app.amount.set(0.5)
        app._amount_changed()
        assert workflow.preview_state is None
        assert str(app.export_button.cget("state")) == "disabled"
        assert all(
            label.cget("text") == "Preview not rendered"
            for label in app.preview_labels.values()
        )

        app._show_error(ProductDesktopError("bounded failure"))
        assert shown[-1] == ("K-MCFM stopped safely", "bounded failure")
        assert app.status.get() == "Stopped safely: bounded failure"

        output = tmp_path / "result.png"
        recipe = tmp_path / "result.recipe.json"
        app._export_complete(
            DesktopExportReceipt(
                style_id="ektar_100",
                look_amount=0.5,
                input_path=source,
                input_sha256="0" * 64,
                output_path=output,
                output_sha256="1" * 64,
                recipe_path=recipe,
                recipe_sha256="2" * 64,
                recipe={},
            )
        )
        assert shown[-1][0] == "Export complete"
        assert "film-inspired / Look Approximation" in shown[-1][1]
        assert output.name in app.status.get()

        second = tmp_path / "second.png"
        _source(second)
        state = workflow.render_previews(source, 0.5)
        target = Path(state.manifest["rows"][0]["output_path"])
        target.unlink()
        target.write_bytes(b"foreign")
        with pytest.raises(ProductDesktopError, match="ownership changed"):
            app._set_input(second)
        assert app.input_path == source.resolve()
        assert app.input_text.get() == source.name
        assert workflow.preview_state is state
        assert str(app.export_button.cget("state")) == "disabled"
        assert target.read_bytes() == b"foreign"
    finally:
        if root.winfo_exists():
            app.close()


def test_native_entrypoint_starts_and_closes_cleanly(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts/open_product_desktop.py"),
            "--input",
            str(source),
            "--scratch-root",
            str(scratch),
            "--smoke-exit-ms",
            "250",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert list(scratch.iterdir()) == []
