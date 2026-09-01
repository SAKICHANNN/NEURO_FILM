from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.product_desktop import (
    DESKTOP_EFFECT_CAPS,
    DesktopFinishingEffects,
    ProductDesktopError,
    ProductDesktopWorkflow,
    sha256_file,
)
from src.inference.product_detail_inspection import render_exact_export_detail

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path, offset: int = 0) -> None:
    yy, xx = np.mgrid[:37, :53]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + offset) % 251,
            (xx * 5 + yy * 13 + 17 + offset) % 251,
            (xx * 7 + yy * 19 + 41 + offset) % 251,
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
        tile_size=64,
        tile_workers=1,
        png_compression=6,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grain": True}, "finite number"),
        ({"grain": float("nan")}, "frozen cap"),
        ({"grain": 0.0500001}, "frozen cap"),
        ({"halation": -0.001}, "frozen cap"),
        ({"dust": 0.020001}, "frozen cap"),
        ({"seed": 8}, "fixed at 7"),
        ({"seed": True}, "fixed at 7"),
    ],
)
def test_effect_value_rejects_noncanonical_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ProductDesktopError, match=message):
        DesktopFinishingEffects(**kwargs)


def test_effect_identity_and_frozen_caps_are_explicit() -> None:
    identity = DesktopFinishingEffects()
    maximum = DesktopFinishingEffects(**DESKTOP_EFFECT_CAPS)
    assert identity.is_identity
    assert not maximum.is_identity
    assert maximum.receipt_identity() == {
        "grain": 0.05,
        "halation": 0.15,
        "dust": 0.02,
        "seed": 7,
        "halation_model": "simple",
    }


def test_zero_effect_command_is_legacy_exact_and_nonzero_is_explicit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    workflow = _workflow(tmp_path)
    try:
        workflow.render_previews(source, 0.65)
        output = tmp_path / "result.png"
        legacy = workflow.export_command("ektar_100", output)
        assert legacy == workflow.export_command(
            "ektar_100", output, effects=DesktopFinishingEffects()
        )
        assert not {"--grain", "--halation", "--dust", "--seed"} & set(legacy)
        maximum = workflow.export_command(
            "ektar_100",
            output,
            effects=DesktopFinishingEffects(**DESKTOP_EFFECT_CAPS),
        )
        for option, expected in (
            ("--grain", 0.05),
            ("--halation", 0.15),
            ("--dust", 0.02),
            ("--seed", 7.0),
        ):
            assert maximum.count(option) == 1
            assert float(maximum[maximum.index(option) + 1]) == expected
    finally:
        workflow.close()


def test_effect_export_recipe_and_exact_detail_match(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    effects = DesktopFinishingEffects(**DESKTOP_EFFECT_CAPS)
    workflow = _workflow(tmp_path)
    try:
        workflow.render_previews(source, 0.65)
        detail = render_exact_export_detail(
            workflow,
            "ektar_100",
            output_format_id="png16",
            point=(0.73, 0.31),
            crop_limit=17,
            effects=effects,
        )
        output = tmp_path / "effect.png"
        receipt = workflow.export(
            "ektar_100",
            output,
            output_format_id="png16",
            effects=effects,
        )
        assert receipt.effects == effects
        assert detail.full_output_sha256 == receipt.output_sha256
        assert receipt.recipe["render"]["effects"] == {
            "grain": {"strength": 0.05, "seed": 7, "color": True},
            "halation": {
                "strength": 0.15,
                "model": "simple",
                "preset": None,
                "control_mode": "locked",
                "resolved_parameters": None,
            },
            "dust": {"strength": 0.02, "seed": 24},
        }
        with Image.open(output) as opened:
            oracle = np.asarray(opened.convert("RGB").crop(detail.crop_box))
        with Image.open(BytesIO(detail.crop_png)) as opened:
            candidate = np.asarray(opened)
        np.testing.assert_array_equal(candidate, oracle)
        output.unlink()
        receipt.recipe_path.unlink()
        assert not tuple((tmp_path / "scratch").glob("u7-16a-detail-*"))
    finally:
        workflow.close()


def test_nonzero_batch_binds_v3_effect_identity(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    _source(first)
    _source(second, 29)
    effects = DesktopFinishingEffects(grain=0.025, halation=0.075, dust=0.01)
    workflow = _workflow(tmp_path)
    destination = tmp_path / "batch"
    try:
        _, inputs = workflow.render_batch_previews((first, second), 0.65)
        receipt = workflow.export_batch(
            inputs,
            "portra_400",
            destination,
            effects=effects,
        )
        assert receipt.effects == effects
        aggregate = json.loads(receipt.receipt_path.read_text("utf-8"))
        assert aggregate["schema_version"] == "kmcfm.desktop-single-look-batch.v3"
        assert aggregate["finishing_effects"] == effects.receipt_identity()
        for row in aggregate["jobs"]:
            recipe = json.loads((destination / row["recipe_path"]).read_text("utf-8"))
            assert recipe["render"]["effects"]["grain"]["strength"] == 0.025
            assert recipe["render"]["effects"]["halation"]["strength"] == 0.075
            assert recipe["render"]["effects"]["dust"]["strength"] == 0.01
            assert sha256_file(destination / row["output_path"]) == row["output_sha256"]
    finally:
        workflow.close()


def test_native_ui_maps_integer_percent_without_invalidating_colour_preview(
    tmp_path: Path,
) -> None:
    import tkinter as tk

    from src.inference.product_desktop_ui import ProductDesktopApp

    workflow = _workflow(tmp_path)
    root = tk.Tk()
    root.withdraw()
    try:
        app = ProductDesktopApp(root, workflow)
        app.preview_ready = True
        app.effect_percents["grain"].set(50)
        app._effect_changed("grain", 50.4)
        app.effect_percents["halation"].set(100)
        app._effect_changed("halation", 100)
        app.effect_percents["dust"].set(25)
        app._effect_changed("dust", 25)
        assert app._finishing_effects() == DesktopFinishingEffects(
            grain=0.025,
            halation=0.15,
            dust=0.005,
        )
        assert app.preview_ready
        assert "Colour previews remain valid" in app.status.get()
        app._set_busy(True, "busy")
        assert all(
            str(control["state"]) == "disabled"
            for control in app.effect_scales.values()
        )
    finally:
        workflow.close()
        root.destroy()


def test_ui_truthfully_labels_colour_only_preview_policy() -> None:
    source = (ROOT / "src/inference/product_desktop_ui.py").read_text("utf-8")
    assert "Colour cards exclude effects" in source
    assert "exact result appears in 1:1 detail" in source
