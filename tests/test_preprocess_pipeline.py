from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image

from src.preprocess import inspect_input, load_working_image, save_srgb16_png, save_srgb16_tiff
from src.preprocess.output_encode import _inject_png_icc
from src.preprocess.raw_decode import load_raw_working_image


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
    assert image.source_transfer_state == "display_referred"
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


def test_profiled_srgb16_tiff_decode_preserves_uint16_precision(tmp_path: Path) -> None:
    rgb = np.linspace(0.0, 1.0, 7 * 9 * 3, dtype=np.float32).reshape(7, 9, 3)
    path = tmp_path / "sample16.tiff"
    save_srgb16_tiff(rgb, path)
    report = inspect_input(path)
    image = load_working_image(path)
    encoded = np.rint(rgb * 65535.0).astype(np.uint16).astype(np.float32) / 65535.0
    expected = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)
    assert report.bit_depth == 16
    assert image.bit_depth_in == 16
    assert image.source_profile.kind == "icc"
    assert np.max(np.abs(image.pixels - expected)) < 1e-7


def test_unknown_profiled_tiff16_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "unknown_profile.tiff"
    array = np.zeros((3, 4, 3), dtype=np.uint16)
    profile = b"not-a-supported-icc-profile"
    tifffile.imwrite(
        path,
        array,
        photometric="rgb",
        metadata=None,
        extratags=[(34675, "B", len(profile), profile, False)],
    )
    with pytest.raises(ValueError, match="ICC conversion is not implemented"):
        load_working_image(path)


def test_unprofiled_tiff16_preserves_precision_with_explicit_srgb_assumption(tmp_path: Path) -> None:
    path = tmp_path / "unprofiled.tiff"
    array = np.arange(3 * 4 * 3, dtype=np.uint16).reshape(3, 4, 3) * 1733
    tifffile.imwrite(path, array, photometric="rgb", metadata=None)
    image = load_working_image(path)
    assert image.bit_depth_in == 16
    assert image.source_profile.kind == "assumed_srgb"
    assert any(warning.code == "assumed_srgb" for warning in image.warnings)
    assert len(np.unique(image.pixels[..., 0])) > 8


def test_profiled_srgb16_png_decode_preserves_uint16_precision(tmp_path: Path) -> None:
    rgb = np.linspace(0.0, 1.0, 7 * 9 * 3, dtype=np.float32).reshape(7, 9, 3)
    path = tmp_path / "sample16.png"
    save_srgb16_png(rgb, path)
    report = inspect_input(path)
    image = load_working_image(path)
    encoded = np.rint(rgb * 65535.0).astype(np.uint16).astype(np.float32) / 65535.0
    expected = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)
    assert report.bit_depth == 16
    assert image.bit_depth_in == 16
    assert image.source_profile.kind == "icc"
    assert np.max(np.abs(image.pixels - expected)) < 1e-7


def test_unprofiled_png16_preserves_precision_with_explicit_srgb_assumption(tmp_path: Path) -> None:
    path = tmp_path / "unprofiled.png"
    rgb = np.arange(3 * 4 * 3, dtype=np.uint16).reshape(3, 4, 3) * 1733
    assert cv2.imwrite(str(path), rgb[..., ::-1])
    image = load_working_image(path)
    assert image.bit_depth_in == 16
    assert image.source_profile.kind == "assumed_srgb"
    assert any(warning.code == "assumed_srgb" for warning in image.warnings)
    assert len(np.unique(image.pixels[..., 0])) > 8


def test_unknown_profiled_png16_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "unknown_profile.png"
    array = np.zeros((3, 4, 3), dtype=np.uint16)
    succeeded, encoded = cv2.imencode(".png", array)
    assert succeeded
    path.write_bytes(_inject_png_icc(encoded.tobytes(), b"not-a-supported-icc-profile"))
    with pytest.raises(ValueError, match="ICC conversion is not implemented"):
        load_working_image(path)


def test_raw_decode_requests_linear_srgb_and_preserves_scene_state(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    from src.preprocess import raw_decode

    calls = []

    class FakeRaw:
        raw_type = "flat"
        color_desc = b"RGBG"
        num_colors = 3
        black_level_per_channel = [0, 0, 0, 0]
        white_level = 16383
        camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
        daylight_whitebalance = [2.0, 1.0, 1.5, 1.0]
        sizes = SimpleNamespace(raw_width=6, raw_height=4, width=4, height=3)

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
    monkeypatch.setattr(raw_decode, "_rawpy", lambda: fake_rawpy)
    path = tmp_path / "sample.dng"
    path.write_bytes(b"fixture")
    working = load_raw_working_image(path)
    assert calls[0]["output_color"] == "explicit-srgb"
    assert calls[0]["gamma"] == (1, 1)
    assert working.working_space == "linear_srgb"
    assert working.transfer_state == "scene_linear"
    assert working.source_transfer_state == "scene_linear"
    assert any(warning.code == "generic_raw_display_mapping" for warning in working.warnings)
