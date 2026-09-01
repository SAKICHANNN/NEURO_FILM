from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest

from scripts import open_product_desktop
from src.inference.product_desktop import ProductDesktopError, ProductDesktopWorkflow

ROOT = Path(__file__).resolve().parents[1]


def _workflow(tmp_path: Path, **kwargs: object) -> ProductDesktopWorkflow:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=2_000,
        tile_size=23,
        tile_workers=1,
        **kwargs,  # type: ignore[arg-type]
    )


def _option(command: tuple[str, ...], name: str) -> str:
    return command[command.index(name) + 1]


def test_legacy_constructor_keeps_one_resource_policy(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"not-decoded")
    workflow = _workflow(tmp_path)

    command = workflow._build_export_command(
        source, "ektar_100", tmp_path / "x.png", 0.65
    )

    assert workflow.tile_size == 23
    assert workflow.tile_workers == 1
    assert workflow.export_tile_size == 23
    assert workflow.export_tile_workers == 1
    assert _option(command, "--tile-size") == "23"
    assert _option(command, "--tile-workers") == "1"


def test_explicit_export_policy_does_not_change_preview_policy(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"not-decoded")
    workflow = _workflow(
        tmp_path,
        export_tile_size=512,
        export_tile_workers=8,
    )

    command = workflow._build_export_command(
        source, "ektar_100", tmp_path / "x.png", 0.65
    )

    assert (workflow.tile_size, workflow.tile_workers) == (23, 1)
    assert (workflow.export_tile_size, workflow.export_tile_workers) == (512, 8)
    assert _option(command, "--tile-size") == "512"
    assert _option(command, "--tile-workers") == "8"
    assert _option(command, "--png-compression") == "6"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"export_tile_size": True},
        {"export_tile_size": 0},
        {"export_tile_size": 1.5},
        {"export_tile_workers": False},
        {"export_tile_workers": 0},
        {"export_tile_workers": "8"},
    ],
)
def test_export_policy_rejects_invalid_values(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    with pytest.raises(ProductDesktopError, match="export_tile"):
        _workflow(tmp_path, **kwargs)


def test_real_desktop_launcher_selects_fast_export_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeRoot:
        def after(self, milliseconds: int, callback: object) -> None:
            captured["after"] = (milliseconds, callback)

        def mainloop(self) -> None:
            captured["mainloop"] = True

    class FakeWorkflow:
        def __init__(self, **kwargs: object) -> None:
            captured["workflow"] = kwargs

    class FakeApp:
        def close(self) -> None:
            captured["closed"] = True

    tkinter = types.ModuleType("tkinter")
    tkinter.Tk = FakeRoot  # type: ignore[attr-defined]
    product_desktop = types.ModuleType("src.inference.product_desktop")
    product_desktop.ProductDesktopWorkflow = FakeWorkflow  # type: ignore[attr-defined]
    product_desktop_ui = types.ModuleType("src.inference.product_desktop_ui")
    product_desktop_ui.build_product_desktop_app = (  # type: ignore[attr-defined]
        lambda root, workflow, initial_input: FakeApp()
    )
    monkeypatch.setitem(sys.modules, "tkinter", tkinter)
    monkeypatch.setitem(sys.modules, "src.inference.product_desktop", product_desktop)
    monkeypatch.setitem(
        sys.modules, "src.inference.product_desktop_ui", product_desktop_ui
    )
    monkeypatch.setattr(
        open_product_desktop,
        "parse_args",
        lambda: argparse.Namespace(
            input=None,
            scratch_root=tmp_path,
            smoke_exit_ms=1,
        ),
    )
    monkeypatch.setattr(
        open_product_desktop,
        "resolve_product_scratch_root",
        lambda candidate: tmp_path,
    )

    assert open_product_desktop.main() == 0
    kwargs = captured["workflow"]
    assert isinstance(kwargs, dict)
    assert kwargs["export_tile_size"] == 512
    assert kwargs["export_tile_workers"] == 8
    assert "tile_size" not in kwargs
    assert "tile_workers" not in kwargs
    assert captured["mainloop"] is True
