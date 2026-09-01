from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference import product_desktop, product_desktop_ui
from src.inference.product_desktop import (
    PRODUCT_PREVIEW_DISPLAY_SIZE,
    ProductDesktopWorkflow,
)
from src.inference.render_contract import sha256_file
from src.inference.three_stock_preview import (
    ThreeStockPreviewError,
    preview_dimensions,
    render_three_stock_previews_to_directory,
)

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path, *, width: int, height: int) -> None:
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


def test_display_box_geometry_is_exact_bounded_and_additive() -> None:
    assert preview_dimensions(4_032, 6_048, 1_000_000) == (816, 1_224)
    assert preview_dimensions(
        4_032, 6_048, 1_000_000, max_width=300, max_height=260
    ) == (173, 260)
    assert preview_dimensions(
        4_128, 2_176, 1_000_000, max_width=300, max_height=260
    ) == (300, 158)
    assert preview_dimensions(
        1_000, 1_000, 1_000_000, max_width=300, max_height=260
    ) == (260, 260)
    assert preview_dimensions(1_600, 200, 1_000_000, max_width=300, max_height=260) == (
        300,
        37,
    )
    assert preview_dimensions(41, 29, 10_000, max_width=300, max_height=260) == (41, 29)


@pytest.mark.parametrize(
    ("max_width", "max_height"),
    [(300, None), (None, 260), (0, 260), (300, -1), (True, 260), (300, 2.5)],
)
def test_display_box_requires_two_positive_integer_bounds(
    max_width: object, max_height: object
) -> None:
    with pytest.raises(
        ThreeStockPreviewError, match="provided together|positive integer"
    ):
        preview_dimensions(
            100,
            80,
            10_000,
            max_width=max_width,  # type: ignore[arg-type]
            max_height=max_height,  # type: ignore[arg-type]
        )


def test_direct_renderer_uses_box_only_when_requested(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source, width=401, height=303)
    common = {
        "root": ROOT,
        "profile_path": ROOT / "configs/render_profiles/safe_rich_v1.json",
        "statistics_path": ROOT / "configs/film_color_stats.json",
        "guardrails_path": ROOT / "configs/color_guardrails.json",
        "max_preview_pixels": 1_000_000,
        "tile_size": 23,
    }
    historical = render_three_stock_previews_to_directory(
        source, tmp_path / "historical", **common
    )
    candidate = render_three_stock_previews_to_directory(
        source,
        tmp_path / "candidate",
        max_preview_width=300,
        max_preview_height=260,
        **common,
    )

    assert (historical["preview_width"], historical["preview_height"]) == (401, 303)
    assert "max_preview_width" not in historical
    assert (candidate["preview_width"], candidate["preview_height"]) == (300, 226)
    assert candidate["max_preview_width"] == 300
    assert candidate["max_preview_height"] == 260
    assert (
        json.loads((tmp_path / "candidate" / "preview.json").read_text("utf-8"))
        == candidate
    )


def test_product_workflow_requests_the_shared_visible_box(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source, width=401, height=303)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    workflow = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        tile_size=23,
    )
    state = workflow.render_previews(source, 0.65)

    assert PRODUCT_PREVIEW_DISPLAY_SIZE == (300, 260)
    assert (
        product_desktop_ui.PRODUCT_PREVIEW_DISPLAY_SIZE
        is product_desktop.PRODUCT_PREVIEW_DISPLAY_SIZE
    )
    assert (state.manifest["preview_width"], state.manifest["preview_height"]) == (
        300,
        226,
    )
    assert state.manifest["max_preview_width"] == 300
    assert state.manifest["max_preview_height"] == 260
    assert all(
        Image.open(Path(str(row["output_path"]))).size == (300, 226)
        for row in state.manifest["rows"]
    )
    assert workflow.close() is True
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize(
    ("manifest_update", "message"),
    [
        ({"preview_width": 301, "preview_pixels": 68_026}, "display bounds drift"),
        ({"max_preview_width": 299}, "display request drift"),
        ({"preview_pixels": 67_799}, "preview geometry drift"),
        ({"preview_width": True}, "preview geometry drift"),
    ],
)
def test_product_workflow_rejects_display_manifest_drift_without_residue(
    tmp_path: Path, manifest_update: dict[str, int], message: str
) -> None:
    source = tmp_path / "source.png"
    _source(source, width=401, height=303)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def drifting_renderer(
        input_path: Path, output_directory: Path, **kwargs: object
    ) -> dict[str, object]:
        assert kwargs["max_preview_width"] == 300
        assert kwargs["max_preview_height"] == 260
        manifest: dict[str, object] = {
            "schema_version": "neuro-film.three-stock-direct-preview.v1",
            "input_sha256": sha256_file(input_path),
            "look_amount": 0.5,
            "preview_width": 300,
            "preview_height": 226,
            "preview_pixels": 67_800,
            "max_preview_width": 300,
            "max_preview_height": 260,
            "rows": [],
        }
        manifest.update(manifest_update)
        return manifest

    workflow = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        preview_renderer=drifting_renderer,
    )
    with pytest.raises(product_desktop.ProductDesktopError, match=message):
        workflow.render_previews(source, 0.5)
    assert workflow.preview_state is None
    assert list(scratch.iterdir()) == []
