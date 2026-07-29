from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.eval.real_uniform_grain_preflight import (
    UniformGrainPreflightError,
    inspect_uniform_grain_tiff,
)


def _write_fixture(path: Path, *, pages: int = 1, samples: int = 4) -> np.ndarray:
    y, x = np.mgrid[:128, :160]
    base = 12000 + 17 * x + 9 * y
    planes = [
        np.asarray(base + 500 * index, dtype=np.uint16)
        for index in range(samples)
    ]
    image = np.stack(planes, axis=-1)
    if pages == 1:
        tifffile.imwrite(
            path,
            image,
            photometric="rgb" if samples in {3, 4} else "minisblack",
            compression=None,
            resolution=(4000, 4000),
            resolutionunit="INCH",
            extrasamples=["UNASSALPHA"] if samples == 4 else None,
        )
    else:
        with tifffile.TiffWriter(path) as writer:
            for _ in range(pages):
                writer.write(
                    image,
                    photometric="rgb",
                    compression=None,
                    resolution=(4000, 4000),
                    resolutionunit="INCH",
                    extrasamples=["UNASSALPHA"],
                )
    return image


def _expected(path: Path, image: np.ndarray) -> dict:
    return {
        "title": "File:KodakTestGrainScan01.tif",
        "film_stock_id": "test-stock",
        "path": str(path),
        "expected_bytes": path.stat().st_size,
        "api_sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
        "width": image.shape[1],
        "height": image.shape[0],
    }


def test_exact_rgba_tiff_keeps_ir_separate(tmp_path: Path) -> None:
    path = tmp_path / "scan.tif"
    image = _write_fixture(path)
    report, rgb, infrared = inspect_uniform_grain_tiff(
        path=path,
        expected=_expected(path, image),
        crop_size=64,
        centers_yx=[[0.3, 0.3], [0.7, 0.7]],
    )
    assert rgb.shape == (128, 160, 3)
    assert infrared.shape == (128, 160)
    assert np.array_equal(rgb, image[..., :3])
    assert np.array_equal(infrared, image[..., 3])
    assert report["rgb_ir_separated"] is True
    assert report["resolution_dpi"] == [4000.0, 4000.0]


def test_preflight_rejects_multipage_or_three_channel_tiff(
    tmp_path: Path,
) -> None:
    path = tmp_path / "multipage.tif"
    image = _write_fixture(path, pages=2)
    with pytest.raises(UniformGrainPreflightError, match="one page"):
        inspect_uniform_grain_tiff(
            path=path,
            expected=_expected(path, image),
            crop_size=64,
            centers_yx=[[0.5, 0.5]],
        )

    path = tmp_path / "rgb.tif"
    image = _write_fixture(path, samples=3)
    with pytest.raises(UniformGrainPreflightError, match="uint16 RGBA"):
        inspect_uniform_grain_tiff(
            path=path,
            expected=_expected(path, image),
            crop_size=64,
            centers_yx=[[0.5, 0.5]],
        )


def test_preflight_rejects_identity_drift(tmp_path: Path) -> None:
    path = tmp_path / "scan.tif"
    image = _write_fixture(path)
    expected = deepcopy(_expected(path, image))
    expected["api_sha1"] = "0" * 40
    with pytest.raises(UniformGrainPreflightError, match="identity"):
        inspect_uniform_grain_tiff(
            path=path,
            expected=expected,
            crop_size=64,
            centers_yx=[[0.5, 0.5]],
        )
