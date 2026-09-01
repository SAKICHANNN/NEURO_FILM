from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from src.inference.product_desktop import ProductDesktopError, ProductDesktopWorkflow
from src.inference.product_desktop_ui import build_product_desktop_app
from src.inference.render_contract import sha256_file
from src.inference.three_stock_preview import (
    ThreeStockPreviewError,
    render_three_stock_previews_to_directory,
)
from src.preprocess import (
    load_working_image,
    save_srgb8,
    working_image_to_srgb_float,
)

ROOT = Path(__file__).resolve().parents[1]
_STYLE_FILES = (
    "velvia_50.preview.png",
    "portra_400.preview.png",
    "ektar_100.preview.png",
)
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


def _source(path: Path, *, width: int = 401, height: int = 303) -> None:
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (xx * 13 + yy * 7) % 251,
            (xx * 3 + yy * 17 + 19) % 251,
            (xx * 11 + yy * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _render(source: Path, destination: Path, **kwargs: object) -> dict[str, Any]:
    return render_three_stock_previews_to_directory(
        source,
        destination,
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        max_preview_pixels=1_000_000,
        max_preview_width=300,
        max_preview_height=260,
        look_amount=0.625,
        seed=7,
        tile_size=23,
        tile_workers=1,
        png_compression=6,
        **kwargs,
    )


def _portable_manifest(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    portable = json.loads(json.dumps(manifest, sort_keys=True))
    for row in portable["rows"]:
        path = Path(row["output_path"])
        assert path.parent == root.resolve()
        row["output_path"] = f"<root>/{path.name}"
    return portable


def test_opt_in_false_preserves_historical_tree_and_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    omitted_root = tmp_path / "omitted"
    explicit_root = tmp_path / "explicit"
    omitted = _render(source, omitted_root)
    explicit = _render(source, explicit_root, include_input_preview=False)

    assert _portable_manifest(omitted, omitted_root) == _portable_manifest(
        explicit, explicit_root
    )
    assert sorted(path.name for path in omitted_root.iterdir()) == [
        "ektar_100.preview.png",
        "portra_400.preview.png",
        "preview.json",
        "velvia_50.preview.png",
    ]
    assert "input_preview" not in omitted
    assert all(
        (omitted_root / name).read_bytes() == (explicit_root / name).read_bytes()
        for name in _STYLE_FILES
    )


def test_opt_in_preview_is_oracle_exact_without_look_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.inference import three_stock_preview

    source = tmp_path / "source.png"
    _source(source)
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline = _render(source, baseline_root)
    decode_calls = 0
    original_loader = three_stock_preview.load_working_image

    def counted_loader(path: Path) -> Any:
        nonlocal decode_calls
        decode_calls += 1
        return original_loader(path)

    monkeypatch.setattr(three_stock_preview, "load_working_image", counted_loader)
    candidate = _render(source, candidate_root, include_input_preview=True)

    assert decode_calls == 1
    assert [row["output_sha256"] for row in candidate["rows"]] == [
        row["output_sha256"] for row in baseline["rows"]
    ]
    assert all(
        (baseline_root / name).read_bytes() == (candidate_root / name).read_bytes()
        for name in _STYLE_FILES
    )

    working = load_working_image(source)
    resized = np.ascontiguousarray(
        cv2.resize(working.pixels, (300, 226), interpolation=cv2.INTER_AREA),
        dtype=np.float32,
    )
    oracle_rgb = working_image_to_srgb_float(replace(working, pixels=resized))
    oracle_path = tmp_path / "oracle.png"
    save_srgb8(oracle_rgb, oracle_path, png_compression=6)
    input_path = candidate_root / "input.preview.png"
    assert input_path.read_bytes() == oracle_path.read_bytes()
    assert candidate["input_preview"] == {
        "role": "input_basis",
        "output_path": str(input_path.resolve()),
        "output_sha256": sha256_file(input_path),
        "source_kind": "raster",
        "display_adapter": "existing WorkingImage to display-sRGB adapter",
        "claim": "generic display adapter, not a calibrated camera rendering",
    }
    assert [row["style_id"] for row in candidate["rows"]] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
    ]


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_input_preview_opt_in_requires_boolean(tmp_path: Path, value: object) -> None:
    source = tmp_path / "source.png"
    _source(source, width=41, height=29)
    with pytest.raises(ThreeStockPreviewError, match="must be a boolean"):
        _render(
            source,
            tmp_path / f"bad-{type(value).__name__}",
            include_input_preview=value,
        )


def test_product_workflow_binds_reads_and_rejects_input_preview_tamper(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    workflow = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        tile_size=23,
    )
    state = workflow.render_previews(source, 0.625)
    input_row = state.manifest["input_preview"]
    input_path = Path(input_row["output_path"])

    assert workflow.input_preview_bytes() == input_path.read_bytes()
    assert input_path in {seal.identity.path for seal in state.files}
    assert len(workflow.preview_bytes()) == 3
    input_path.write_bytes(b"foreign-tamper")
    with pytest.raises(ProductDesktopError, match="workspace ownership changed"):
        workflow.input_preview_bytes()
    assert workflow.close() is False
    assert input_path.read_bytes() == b"foreign-tamper"


def test_native_ui_shows_truthful_nonselectable_input_basis_and_clears_it(
    tmp_path: Path,
) -> None:
    import tkinter as tk

    source = tmp_path / "source.png"
    _source(source)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    workflow = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        tile_size=23,
    )
    root = tk.Toplevel(_shared_tk_root())
    root.withdraw()
    app = build_product_desktop_app(root, workflow, initial_input=source)
    try:
        state = workflow.render_previews(source, 0.625)
        app._preview_complete(state)
        assert app.input_preview_image is not None
        assert app.input_preview_image.width() <= 160
        assert app.input_preview_image.height() <= 120
        assert str(app.input_preview_label.cget("text")) == ""
        assert "not a calibrated camera rendering" in str(
            app.input_preview_caption.cget("text")
        )
        assert tuple(app.look_buttons) == (
            "velvia_50",
            "portra_400",
            "ektar_100",
        )
        assert app.style.get() == ""
        assert str(app.export_button.cget("state")) == "disabled"

        app.amount.set(0.5)
        app._amount_changed()
        assert app.input_preview_image is None
        assert app.input_preview_label.cget("text") == "Input basis not rendered"
        assert workflow.preview_state is None
    finally:
        if root.winfo_exists():
            app.close()
    assert list(scratch.iterdir()) == []
