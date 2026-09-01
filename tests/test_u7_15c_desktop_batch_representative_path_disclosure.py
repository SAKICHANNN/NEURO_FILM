from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.product_desktop import ProductDesktopError, ProductDesktopWorkflow
from src.inference.product_desktop_ui import build_product_desktop_app

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u7_15c_desktop_batch_representative_path_disclosure_v1.json"
AUTOMATIC = "Automatic · canonical first"


def _source(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:19, :23]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + offset) % 251,
            (xx * 5 + yy * 13 + offset * 2) % 251,
            (xx * 7 + yy * 17 + offset * 3) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _workflow(tmp_path: Path) -> ProductDesktopWorkflow:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=2_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
    )


def test_contract_freezes_basename_only_disclosure_repair() -> None:
    contract = json.loads(CONTRACT.read_text("utf-8"))
    assert contract["parent_head"] == "020bb364d9d86824b8af4bb22418bb7a1e27c9ea"
    assert contract["presentation"] == {
        "automatic_label": AUTOMATIC,
        "explicit_template": "{one_based_index:03d} · {basename}",
        "absolute_parent_visible": False,
        "duplicate_basename_disambiguator": "one_based_index",
        "internal_exact_path_binding": True,
    }
    assert "basename-only" in contract["claim_ceiling"]


def test_duplicate_basenames_remain_exact_without_visible_parent_paths(
    tmp_path: Path,
) -> None:
    import tkinter as tk

    first = tmp_path / "private-alpha" / "same-name.png"
    second = tmp_path / "private-beta" / "same-name.png"
    _source(first, 1)
    _source(second, 2)
    workflow = _workflow(tmp_path)
    root = tk.Tk()
    root.withdraw()
    app = build_product_desktop_app(root, workflow)
    try:
        app._set_inputs((second, first))
        values = tuple(str(value) for value in app.representative_combo.cget("values"))
        assert values == (
            AUTOMATIC,
            "001 · same-name.png",
            "002 · same-name.png",
        )
        visible = "\n".join(
            (*values, app.input_text.get(), app.status.get())
        )
        assert str(first.parent) not in visible
        assert str(second.parent) not in visible
        assert str(tmp_path.resolve()) not in visible

        app.representative.set(values[1])
        assert app._selected_representative_path() == second.resolve()
        app._representative_changed()
        assert app.input_path == second.resolve()
        assert app.input_text.get() == (
            "2 photos · preview representative same-name.png"
        )

        app.representative.set(values[2])
        assert app._selected_representative_path() == first.resolve()
        app._representative_changed()
        assert app.input_path == first.resolve()
        assert str(first.parent) not in app.input_text.get()
        assert str(second.parent) not in app.status.get()

        app.representative.set(str(first))
        with pytest.raises(
            ProductDesktopError, match="batch preview representative is unavailable"
        ):
            app._selected_representative_path()

        app._set_inputs((first,))
        assert tuple(app.representative_combo.cget("values")) == ()
        assert str(app.representative_combo.cget("state")) == "disabled"
        assert app.input_text.get() == first.name
    finally:
        if root.winfo_exists():
            app.close()


def test_source_uses_basename_not_complete_path() -> None:
    source = (ROOT / "src/inference/product_desktop_ui.py").read_text("utf-8")
    assert 'label = f"{index:03d} · {path.name}"' in source
    assert 'label = f"{index:03d} · {path}"' not in source
