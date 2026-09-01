from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.inference.product_desktop import (
    DesktopExportReceipt,
    ProductDesktopError,
    ProductDesktopWorkflow,
    product_output_format,
    sha256_file,
)
from src.inference.product_detail_inspection import (
    DesktopDetailPreview,
    detail_crop_box,
    normalized_card_point,
    render_exact_export_detail,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _FakeWorkflow:
    scratch_root: Path
    extra_member: bool = False
    wrong_hash: bool = False

    def export(
        self,
        style_id: str,
        output_path: Path,
        *,
        output_format_id: str,
    ) -> DesktopExportReceipt:
        spec = product_output_format(output_format_id)
        yy, xx = np.mgrid[:23, :31]
        rgb = np.stack(
            (
                (xx * 9 + yy * 3) % 251,
                (xx * 5 + yy * 11 + 17) % 251,
                (xx * 13 + yy * 7 + 41) % 251,
            ),
            axis=-1,
        ).astype(np.uint8)
        image = Image.fromarray(rgb, mode="RGB")
        if spec.recipe_format == "JPEG":
            image.save(output_path, format="JPEG", quality=95, subsampling=0)
        elif spec.recipe_format == "TIFF":
            image.save(output_path, format="TIFF", compression="raw")
        else:
            image.save(output_path, format="PNG", compress_level=6)
        recipe_path = output_path.with_suffix(".recipe.json")
        recipe_path.write_text(
            json.dumps({"style": style_id, "format": output_format_id}),
            encoding="utf-8",
        )
        if self.extra_member:
            (output_path.parent / "foreign.bin").write_bytes(b"foreign")
        output_sha = sha256_file(output_path)
        if self.wrong_hash:
            output_sha = "0" * 64
        return DesktopExportReceipt(
            style_id=style_id,
            look_amount=0.65,
            input_path=self.scratch_root / "same-name.png",
            input_sha256="1" * 64,
            output_path=output_path,
            output_sha256=output_sha,
            recipe_path=recipe_path,
            recipe_sha256=sha256_file(recipe_path),
            recipe={},
            output_format_id=output_format_id,
        )


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.5, 0.5), (9, 5, 22, 18)),
        ((0.0, 0.0), (0, 0, 13, 13)),
        ((1.0, 1.0), (18, 10, 31, 23)),
    ],
)
def test_detail_crop_box_clamps_without_upscale(
    point: tuple[float, float], expected: tuple[int, int, int, int]
) -> None:
    assert detail_crop_box(31, 23, point, crop_limit=13) == expected
    assert detail_crop_box(7, 5, point, crop_limit=13) == (0, 0, 7, 5)


def test_product_session_binds_detail_helper() -> None:
    source = (ROOT / "src/inference/product_desktop.py").read_text("utf-8")
    assert '"src/inference/product_detail_inspection.py"' in source


def test_detail_helper_uses_exact_export_not_preview_bytes() -> None:
    source = (ROOT / "src/inference/product_detail_inspection.py").read_text("utf-8")
    assert "workflow.export(" in source
    assert "preview_bytes" not in source
    assert "render_previews" not in source


def test_normalized_card_point_rejects_padding_and_maps_pixels() -> None:
    assert normalized_card_point(
        10,
        7,
        widget_width=24,
        widget_height=18,
        image_width=20,
        image_height=10,
    ) == pytest.approx((8 / 19, 3 / 9))
    with pytest.raises(ProductDesktopError, match="inside the rendered preview"):
        normalized_card_point(
            1,
            1,
            widget_width=24,
            widget_height=18,
            image_width=20,
            image_height=10,
        )


@pytest.mark.parametrize(
    "point",
    [(-0.1, 0.5), (1.1, 0.5), (float("nan"), 0.5), (True, 0.5)],
)
def test_detail_point_rejects_invalid_values(point: tuple[object, object]) -> None:
    with pytest.raises(ProductDesktopError, match="finite number"):
        detail_crop_box(31, 23, point, crop_limit=13)


@pytest.mark.parametrize("crop_limit", [0, -1, True, 3.5])
def test_detail_crop_limit_rejects_invalid_values(crop_limit: object) -> None:
    with pytest.raises(ProductDesktopError, match="positive integers"):
        detail_crop_box(31, 23, (0.5, 0.5), crop_limit=crop_limit)


@pytest.mark.parametrize("output_format_id", ["png16", "tiff16", "jpeg8"])
def test_exact_export_detail_uses_output_and_removes_owned_pair(
    tmp_path: Path, output_format_id: str
) -> None:
    workflow = _FakeWorkflow(tmp_path)
    result = render_exact_export_detail(
        workflow,
        "ektar_100",
        output_format_id=output_format_id,
        point=(0.73, 0.31),
        crop_limit=13,
    )
    assert result.style_id == "ektar_100"
    assert result.look_amount == 0.65
    assert result.output_format_id == output_format_id
    assert (result.full_width, result.full_height) == (31, 23)
    assert result.crop_box == (16, 1, 29, 14)
    assert hashlib.sha256(result.crop_png).hexdigest() == result.crop_sha256
    with Image.open(BytesIO(result.crop_png)) as crop:
        assert crop.size == (13, 13)
        assert crop.mode == "RGB"
    assert not tuple(tmp_path.glob("u7-16a-detail-*"))


def test_foreign_member_is_preserved_and_fails_closed(tmp_path: Path) -> None:
    workflow = _FakeWorkflow(tmp_path, extra_member=True)
    with pytest.raises(ProductDesktopError, match="member set changed"):
        render_exact_export_detail(
            workflow,
            "ektar_100",
            output_format_id="png16",
        )
    roots = tuple(tmp_path.glob("u7-16a-detail-*"))
    assert len(roots) == 1
    assert (roots[0] / "foreign.bin").read_bytes() == b"foreign"
    assert (roots[0] / "detail.png").is_file()
    assert (roots[0] / "detail.recipe.json").is_file()


def test_receipt_identity_drift_is_preserved_and_fails_closed(tmp_path: Path) -> None:
    workflow = _FakeWorkflow(tmp_path, wrong_hash=True)
    with pytest.raises(ProductDesktopError, match="receipt drifted"):
        render_exact_export_detail(
            workflow,
            "ektar_100",
            output_format_id="png16",
        )
    roots = tuple(tmp_path.glob("u7-16a-detail-*"))
    assert len(roots) == 1
    assert (roots[0] / "detail.png").is_file()


def _source(path: Path) -> None:
    yy, xx = np.mgrid[:31, :47]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3) % 251,
            (xx * 5 + yy * 13 + 17) % 251,
            (xx * 7 + yy * 19 + 41) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _real_workflow(tmp_path: Path) -> ProductDesktopWorkflow:
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


def test_real_detail_export_matches_final_output_and_crop(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    workflow = _real_workflow(tmp_path)
    try:
        workflow.render_previews(source, 0.65)
        detail = render_exact_export_detail(
            workflow,
            "ektar_100",
            output_format_id="png16",
            point=(0.73, 0.31),
            crop_limit=13,
        )
        final_path = tmp_path / "final.png"
        receipt = workflow.export(
            "ektar_100",
            final_path,
            output_format_id="png16",
        )
        assert detail.full_output_sha256 == receipt.output_sha256
        assert detail.full_output_sha256 == sha256_file(final_path)
        with Image.open(final_path) as opened:
            oracle = np.asarray(opened.convert("RGB").crop(detail.crop_box))
        with Image.open(BytesIO(detail.crop_png)) as opened:
            candidate = np.asarray(opened)
        np.testing.assert_array_equal(candidate, oracle)
        final_path.unlink()
        receipt.recipe_path.unlink()
        assert not tuple((tmp_path / "scratch").glob("u7-16a-detail-*"))
    finally:
        workflow.close()


def test_native_ui_exposes_and_clears_exact_detail_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tkinter as tk

    from src.inference import product_desktop_ui

    source = tmp_path / "source.png"
    _source(source)
    workflow = _real_workflow(tmp_path)
    crop_buffer = BytesIO()
    Image.new("RGB", (17, 13), (41, 73, 109)).save(crop_buffer, format="PNG")
    crop_png = crop_buffer.getvalue()
    captured: dict[str, object] = {}

    def fake_detail(
        _workflow: ProductDesktopWorkflow,
        style_id: str,
        *,
        output_format_id: str,
        point: tuple[float, float],
        crop_limit: int,
    ) -> DesktopDetailPreview:
        captured.update(
            style_id=style_id,
            output_format_id=output_format_id,
            point=point,
            crop_limit=crop_limit,
        )
        return DesktopDetailPreview(
            style_id=style_id,
            look_amount=0.65,
            input_path=source,
            input_sha256=sha256_file(source),
            output_format_id=output_format_id,
            full_width=47,
            full_height=31,
            crop_box=(15, 9, 32, 22),
            crop_png=crop_png,
            crop_sha256=hashlib.sha256(crop_png).hexdigest(),
            full_output_sha256="2" * 64,
            strict_recipe_sha256="3" * 64,
        )

    monkeypatch.setattr(product_desktop_ui, "render_exact_export_detail", fake_detail)
    root = tk.Tk()
    root.geometry("1080x720")
    root.withdraw()
    app = product_desktop_ui.build_product_desktop_app(root, workflow)
    try:
        app._set_input(source)
        state = workflow.render_previews(source, 0.65)
        app._preview_complete(state)
        root.deiconify()
        root.update()
        label = app.preview_labels["ektar_100"]
        photo = app.preview_images[2]
        left = (label.winfo_width() - photo.width()) // 2
        top = (label.winfo_height() - photo.height()) // 2
        app._detail_point_selected(
            "ektar_100",
            SimpleNamespace(
                x=left + photo.width() // 2,
                y=top + photo.height() // 2,
            ),
        )
        assert app.style.get() == "ektar_100"
        assert app.detail_point == pytest.approx((0.5, 0.5), abs=0.03)
        assert str(app.detail_button.cget("state")) == "normal"
        assert app.preview_labels["ektar_100"].bind("<Button-1>")
        app.render_detail()
        deadline = time.monotonic() + 5.0
        while app._foreground_thread is not None and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        assert app._foreground_thread is None
        assert app.detail_window is not None
        assert app.detail_window.winfo_exists()
        assert captured == {
            "style_id": "ektar_100",
            "output_format_id": "png16",
            "point": (0.5, 0.5),
            "crop_limit": 512,
        }
        app.output_format.set("jpeg8")
        app._output_format_changed()
        assert app.detail_window is None
    finally:
        if root.winfo_exists():
            app.close()
