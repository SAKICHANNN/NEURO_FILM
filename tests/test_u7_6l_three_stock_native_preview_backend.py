from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.three_stock_native_preview import (
    NativeThreeStockPreviewBackend,
    build_native_three_stock_preview_backend,
)
from src.inference.three_stock_preview import render_three_stock_previews_to_directory

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path, width: int = 121, height: int = 83) -> None:
    y, x = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _render(
    source: Path,
    output: Path,
    *,
    native_backend: NativeThreeStockPreviewBackend | None = None,
) -> dict:
    return render_three_stock_previews_to_directory(
        source,
        output,
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        max_preview_pixels=4_000,
        look_amount=1.0,
        seed=31,
        tile_size=23,
        tile_workers=2,
        native_backend=native_backend,
    )


def test_opt_in_native_backend_preserves_three_preview_files_exactly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    python_manifest = _render(source, tmp_path / "python")
    assert "color_backend" not in python_manifest

    build_directory = tmp_path / "native-build"
    backend = build_native_three_stock_preview_backend(
        root=ROOT, output_directory=build_directory, thread_count=4
    )
    try:
        native_manifest = _render(
            source, tmp_path / "native", native_backend=backend
        )
    finally:
        backend.close()

    assert [row["style_id"] for row in native_manifest["rows"]] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
    ]
    assert [row["output_sha256"] for row in native_manifest["rows"]] == [
        row["output_sha256"] for row in python_manifest["rows"]
    ]
    assert native_manifest["color_backend"] == {
        "backend_id": "native-safe-lab-pointwise-f32-v3",
        "dll_sha256": backend.dll_sha256,
        "thread_count": 4,
        "gamut_workers": 1,
    }
    assert json.loads((tmp_path / "native" / "preview.json").read_text("utf-8")) == native_manifest


def test_native_backend_failure_leaves_no_publication_or_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    destination = tmp_path / "native"

    def fail(*args, **kwargs):
        raise RuntimeError("injected native failure")
        yield

    monkeypatch.setattr(
        "src.inference.three_stock_native_preview.iter_three_stock_look_rgb_native",
        fail,
    )
    fake_backend = NativeThreeStockPreviewBackend(
        library=object(),
        dll_sha256="0" * 64,
        source_sha256="1" * 64,
        header_sha256="2" * 64,
        toolchain="test",
    )
    with pytest.raises(RuntimeError, match="injected native failure"):
        _render(source, destination, native_backend=fake_backend)
    assert not destination.exists()
    assert list(tmp_path.glob(".native.*.stage")) == []


def test_native_backend_rejects_invalid_resource_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="thread_count"):
        build_native_three_stock_preview_backend(
            root=ROOT, output_directory=tmp_path / "build", thread_count=0
        )
