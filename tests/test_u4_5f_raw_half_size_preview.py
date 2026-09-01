from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.inference import three_stock_preview
from src.inference.three_stock_preview import (
    ThreeStockPreviewError,
    render_three_stock_previews_to_directory,
)
from src.preprocess import raw_preview_decode
from src.preprocess.types import InputInspection, SourceProfile, WorkingImage

ROOT = Path(__file__).resolve().parents[1]


def _inspection(
    path: Path, *, width: int = 4000, height: int = 3000
) -> InputInspection:
    return InputInspection(
        path=path,
        exists=True,
        source_kind="raw",
        format_name="dng",
        width=width,
        height=height,
        bit_depth=16,
        source_profile=SourceProfile("raw_metadata", "LibRaw camera metadata"),
        transfer_state="scene_linear",
    )


def _working(path: Path, *, width: int = 2000, height: int = 1500) -> WorkingImage:
    return WorkingImage(
        pixels=np.full((height, width, 3), 0.25, dtype=np.float32),
        working_space="linear_srgb",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "LibRaw camera metadata"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=path,
        warnings=[],
    )


def test_raw_preview_decoder_requests_native_half_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def postprocess(self, **kwargs):
            calls.append(kwargs)
            return np.full((3, 4, 3), 32768, dtype=np.uint16)

    fake_rawpy = SimpleNamespace(
        ColorSpace=SimpleNamespace(sRGB="explicit-srgb"),
        imread=lambda _path: FakeRaw(),
    )
    source = tmp_path / "source.dng"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(raw_preview_decode, "_rawpy", lambda: fake_rawpy)
    monkeypatch.setattr(
        raw_preview_decode, "inspect_raw", lambda path: _inspection(path)
    )

    working = raw_preview_decode.load_raw_preview_working_image(source)

    assert calls == [
        {
            "use_camera_wb": True,
            "no_auto_bright": True,
            "output_bps": 16,
            "output_color": "explicit-srgb",
            "gamma": (1, 1),
            "user_flip": None,
            "half_size": True,
        }
    ]
    assert working.pixels.shape == (3, 4, 3)
    assert working.pixels.dtype == np.float32
    assert working.pixels.flags.owndata
    assert any(
        warning.code == "raw_half_size_preview_decode" for warning in working.warnings
    )


def test_three_stock_preview_uses_opt_in_raw_half_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.dng"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        three_stock_preview, "inspect_input", lambda path: _inspection(path)
    )
    monkeypatch.setattr(
        three_stock_preview,
        "load_raw_preview_working_image",
        lambda path: _working(path),
    )
    monkeypatch.setattr(
        three_stock_preview,
        "load_working_image",
        lambda _path: pytest.fail("full RAW decode must not run"),
    )

    manifest = render_three_stock_previews_to_directory(
        source,
        tmp_path / "preview",
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        max_preview_pixels=1_000_000,
        tile_size=64,
        tile_workers=1,
        raw_half_size_decode=True,
    )

    assert manifest["raw_half_size_decode"] is True
    assert manifest["jpeg_scaled_decode"] is False
    assert manifest["decoded_width"] == 2000
    assert manifest["decoded_height"] == 1500
    assert manifest["decoded_width"] >= manifest["preview_width"]
    assert manifest["decoded_height"] >= manifest["preview_height"]
    assert manifest["preview_basis"].startswith("LibRaw half-size demosaic")


def test_raw_half_size_preview_rejects_ambiguous_or_unsafe_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.dng"
    source.write_bytes(b"fixture")
    common = {
        "root": ROOT,
        "profile_path": ROOT / "configs/render_profiles/safe_rich_v1.json",
        "statistics_path": ROOT / "configs/film_color_stats.json",
        "guardrails_path": ROOT / "configs/color_guardrails.json",
        "max_preview_pixels": 1_000_000,
    }
    with pytest.raises(ThreeStockPreviewError, match="mutually exclusive"):
        render_three_stock_previews_to_directory(
            source,
            tmp_path / "ambiguous",
            jpeg_scaled_decode=True,
            raw_half_size_decode=True,
            **common,
        )

    png = tmp_path / "source.png"
    Image.new("RGB", (16, 16), (20, 40, 60)).save(png)
    with pytest.raises(ThreeStockPreviewError, match="requires a RAW input"):
        render_three_stock_previews_to_directory(
            png,
            tmp_path / "nonraw",
            max_preview_pixels=64,
            raw_half_size_decode=True,
            **{
                key: value
                for key, value in common.items()
                if key != "max_preview_pixels"
            },
        )

    monkeypatch.setattr(
        three_stock_preview, "inspect_input", lambda path: _inspection(path)
    )
    monkeypatch.setattr(
        three_stock_preview,
        "load_raw_preview_working_image",
        lambda path: _working(path, width=800, height=600),
    )
    with pytest.raises(ThreeStockPreviewError, match="smaller than"):
        render_three_stock_previews_to_directory(
            source,
            tmp_path / "undersized",
            raw_half_size_decode=True,
            **common,
        )
