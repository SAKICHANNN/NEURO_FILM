from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocess import inspect_input, load_working_image


def _rgb_fixture(path: Path) -> None:
    arr = np.zeros((8, 10, 3), dtype=np.uint8)
    arr[..., 0] = 128
    arr[..., 1] = np.arange(10, dtype=np.uint8)[None, :] * 20
    arr[..., 2] = np.arange(8, dtype=np.uint8)[:, None] * 24
    Image.fromarray(arr, mode="RGB").save(path)


def test_inspect_missing_file_reports_structured_warning(tmp_path: Path) -> None:
    report = inspect_input(tmp_path / "missing.jpg")
    assert not report.exists
    assert report.source_kind == "unknown"
    assert report.warnings[0].code == "missing_file"


def test_inspect_png_reports_raster_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    _rgb_fixture(path)
    report = inspect_input(path)
    assert report.exists
    assert report.source_kind == "raster"
    assert report.format_name == "PNG"
    assert report.width == 10
    assert report.height == 8
    assert report.bit_depth == 8
    assert report.source_profile.kind == "assumed_srgb"
    assert any(warning.code == "assumed_srgb" for warning in report.warnings)


def test_load_raster_working_image_is_float32_linear(tmp_path: Path) -> None:
    path = tmp_path / "sample.jpg"
    _rgb_fixture(path)
    image = load_working_image(path)
    assert image.pixels.dtype == np.float32
    assert image.pixels.shape == (8, 10, 3)
    assert image.working_space == "linear_srgb"
    assert image.transfer_state == "display_linear"
    assert 0.0 <= float(image.pixels.min()) <= float(image.pixels.max()) <= 1.0
    assert any(warning.code == "assumed_srgb" for warning in image.warnings)


def test_tiff_inspection_and_decode(tmp_path: Path) -> None:
    path = tmp_path / "sample.tiff"
    _rgb_fixture(path)
    report = inspect_input(path)
    image = load_working_image(path)
    assert report.format_name == "TIFF"
    assert image.bit_depth_in == 8
    assert image.alpha_policy == "absent"
