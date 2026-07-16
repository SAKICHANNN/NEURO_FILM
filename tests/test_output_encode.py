from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image

from src.preprocess import (
    normalized_icc_profile_sha256,
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
    srgb_icc_profile,
    srgb_icc_profile_sha256,
)


@pytest.mark.parametrize(
    ("suffix", "expected_format"),
    [(".png", "PNG"), (".jpg", "JPEG"), (".tiff", "TIFF")],
)
def test_save_srgb8_encoding_matches_extension_and_embeds_icc(
    tmp_path: Path, suffix: str, expected_format: str
) -> None:
    rgb = np.linspace(0.0, 1.0, 8 * 10 * 3, dtype=np.float32).reshape(8, 10, 3)
    path = tmp_path / f"render{suffix}"
    assert save_srgb8(rgb, path) == expected_format
    with Image.open(path) as image:
        assert image.format == expected_format
        assert image.mode == "RGB"
        assert image.size == (10, 8)
        profile = image.info.get("icc_profile", b"")
        assert len(profile) > 0
        assert hashlib.sha256(profile).hexdigest() == srgb_icc_profile_sha256()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_save_srgb8_rejects_unsupported_extension_and_nonfinite_values(tmp_path: Path) -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported output extension"):
        save_srgb8(rgb, tmp_path / "render.webp")
    rgb[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_srgb8(rgb, tmp_path / "render.png")


def test_save_srgb16_tiff_round_trips_uint16_pixels_and_icc(tmp_path: Path) -> None:
    rgb = np.linspace(0.0, 1.0, 8 * 10 * 3, dtype=np.float32).reshape(8, 10, 3)
    path = tmp_path / "render.tiff"
    assert save_srgb16_tiff(rgb, path) == "TIFF"
    expected = np.rint(rgb * 65535.0).astype(np.uint16)
    assert np.array_equal(tifffile.imread(path), expected)
    with tifffile.TiffFile(path) as image:
        profile = bytes(image.pages[0].tags[34675].value)
    assert hashlib.sha256(profile).hexdigest() == srgb_icc_profile_sha256()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_save_srgb16_tiff_rejects_non_tiff_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a .tif"):
        save_srgb16_tiff(np.zeros((2, 3, 3), dtype=np.float32), tmp_path / "render.png")


def test_save_srgb16_png_round_trips_uint16_pixels_and_icc(tmp_path: Path) -> None:
    rgb = np.linspace(0.0, 1.0, 8 * 10 * 3, dtype=np.float32).reshape(8, 10, 3)
    path = tmp_path / "render.png"
    assert save_srgb16_png(rgb, path) == "PNG"
    expected = np.rint(rgb * 65535.0).astype(np.uint16)
    decoded_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert decoded_bgr is not None
    assert decoded_bgr.dtype == np.uint16
    assert np.array_equal(decoded_bgr[..., ::-1], expected)
    with Image.open(path) as image:
        profile = image.info.get("icc_profile", b"")
    assert hashlib.sha256(profile).hexdigest() == srgb_icc_profile_sha256()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_save_srgb16_png_rejects_non_png_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a .png"):
        save_srgb16_png(np.zeros((2, 3, 3), dtype=np.float32), tmp_path / "render.tiff")


def test_normalized_icc_fingerprint_ignores_creation_time_and_profile_id() -> None:
    original = srgb_icc_profile()
    changed = bytearray(original)
    changed[24:36] = bytes(range(12))
    changed[84:100] = bytes(range(16))
    assert hashlib.sha256(changed).hexdigest() != hashlib.sha256(original).hexdigest()
    assert normalized_icc_profile_sha256(changed) == normalized_icc_profile_sha256(original)
