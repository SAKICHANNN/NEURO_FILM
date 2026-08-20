from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from src.eval.repid_shared_operator import load_jpeg_as_srgb


def _jpeg(path: Path) -> tuple[str, str]:
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    values = np.arange(24 * 18 * 3, dtype=np.uint16).reshape(18, 24, 3) % 256
    Image.fromarray(values.astype(np.uint8), mode="RGB").save(
        path,
        format="JPEG",
        quality=95,
        icc_profile=profile,
    )
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), hashlib.sha256(profile).hexdigest()


def test_load_jpeg_as_srgb_is_exact_and_finite(tmp_path: Path) -> None:
    path = tmp_path / "input.jpeg"
    sha256, icc_sha256 = _jpeg(path)
    first = load_jpeg_as_srgb(
        path,
        expected_sha256=sha256,
        expected_icc_sha256=icc_sha256,
    )
    second = load_jpeg_as_srgb(
        path,
        expected_sha256=sha256,
        expected_icc_sha256=icc_sha256,
    )
    assert first.shape == (18, 24, 3)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert np.min(first) >= 0.0 and np.max(first) <= 1.0


def test_load_jpeg_as_srgb_rejects_icc_drift(tmp_path: Path) -> None:
    path = tmp_path / "input.jpeg"
    sha256, _ = _jpeg(path)
    try:
        load_jpeg_as_srgb(path, expected_sha256=sha256, expected_icc_sha256="0" * 64)
    except ValueError as error:
        assert "ICC identity drift" in str(error)
    else:
        raise AssertionError("ICC drift was not rejected")
