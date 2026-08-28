from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imagecodecs
import numpy as np
import pytest
import tifffile
from PIL import Image

from scripts.audit_u1_2d_high_precision_tiff_orientation import run
from src.preprocess import load_working_image
from src.preprocess.output_encode import srgb_icc_profile
from src.preprocess.prophoto_icc import decode_prophoto_rgb16_to_linear_rec2020

ROOT = Path(__file__).resolve().parents[1]

_PIL_ORIENTATIONS = {
    1: None,
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def _fixture() -> np.ndarray:
    positions = np.arange(15, dtype=np.uint16).reshape(3, 5)
    return np.stack(
        (
            positions * 101 + 17,
            positions * 211 + 1003,
            positions * 307 + 5001,
        ),
        axis=2,
    )


def _pillow_oracle(array: np.ndarray, orientation: int) -> np.ndarray:
    operation = _PIL_ORIENTATIONS[orientation]
    if operation is None:
        return array.copy()
    channels = []
    for channel in range(3):
        image = Image.fromarray(array[..., channel])
        channels.append(np.asarray(image.transpose(operation), dtype=np.uint16))
    return np.stack(channels, axis=2)


def _write_tiff(
    path: Path,
    array: np.ndarray,
    orientation: int,
    *,
    profile: bytes | None = None,
) -> None:
    tags: list[tuple[int, str, int, object, bool]] = [
        (274, "H", 1, orientation, False)
    ]
    if profile is not None:
        tags.append((34675, "B", len(profile), profile, False))
    tifffile.imwrite(
        path,
        array,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        extratags=tags,
    )


def _srgb_linear(array: np.ndarray) -> np.ndarray:
    encoded = array.astype(np.float32) / 65535.0
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_all_tiff_orientations_match_independent_pillow_oracle(
    tmp_path: Path, orientation: int
) -> None:
    source = _fixture()
    path = tmp_path / f"orientation_{orientation}.tiff"
    _write_tiff(path, source, orientation, profile=srgb_icc_profile())

    working = load_working_image(path)
    expected_samples = _pillow_oracle(source, orientation)

    assert working.pixels.tobytes() == _srgb_linear(expected_samples).tobytes()
    assert working.pixels.shape == expected_samples.shape
    assert np.array_equal(
        np.sort(expected_samples.reshape(-1, 3), axis=0),
        np.sort(source.reshape(-1, 3), axis=0),
    )
    assert working.orientation_applied
    assert working.bit_depth_in == 16


def _prophoto_profile() -> bytes:
    return imagecodecs.cms_profile(
        "rgb",
        whitepoint=(0.3457, 0.3585, 1.0),
        primaries=(0.7347, 0.2653, 0.1596, 0.8404, 0.0366, 0.0001),
        gamma=1.8,
    )


def test_prophoto_orientation_precedes_colour_transform(tmp_path: Path) -> None:
    source = _fixture()
    profile = _prophoto_profile()
    path = tmp_path / "prophoto_orientation_6.tiff"
    _write_tiff(path, source, 6, profile=profile)

    working = load_working_image(path)
    expected_samples = _pillow_oracle(source, 6)
    expected = decode_prophoto_rgb16_to_linear_rec2020(expected_samples, profile)

    assert working.working_space == "linear_rec2020"
    assert working.pixels.tobytes() == expected.tobytes()


def test_invalid_tiff_orientation_rejects(tmp_path: Path) -> None:
    path = tmp_path / "invalid_orientation.tiff"
    _write_tiff(path, _fixture(), 9, profile=srgb_icc_profile())
    with pytest.raises(ValueError, match="invalid TIFF Orientation value: 9"):
        load_working_image(path)


def test_render_film_consumes_rotated_rgb16_tiff(tmp_path: Path) -> None:
    path = tmp_path / "orientation_6.tiff"
    output = tmp_path / "rendered.png"
    _write_tiff(path, _fixture(), 6, profile=srgb_icc_profile())

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    with Image.open(output) as rendered:
        assert rendered.size == (3, 5)


def test_formal_audit_passes_and_cleans_scratch(tmp_path: Path) -> None:
    report = run(
        ROOT / "configs/u1_2d_high_precision_tiff_orientation_v1.json",
        tmp_path / "report.json",
    )
    assert report["status"] == "PASS_RGB16_TIFF_ORIENTATION_INGRESS"
    assert all(report["gates"].values())
