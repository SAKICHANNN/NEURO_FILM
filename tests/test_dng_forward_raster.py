from __future__ import annotations

import numpy as np
import pytest

from src.preprocess.dng_forward_raster import (
    DngForwardRasterError,
    _apply_camera_to_rec2020,
    load_dng_forward_working_image,
)


def test_camera_raster_conversion_is_float32_finite_and_does_not_mutate() -> None:
    camera = np.asarray([[[0, 32768, 65535], [1000, 2000, 3000]]], dtype=np.uint16)
    frozen = camera.copy()
    output = _apply_camera_to_rec2020(camera, np.eye(3), row_block=1)
    assert output.shape == camera.shape
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))
    assert np.array_equal(camera, frozen)


@pytest.mark.parametrize(
    ("camera", "matrix", "row_block", "match"),
    [
        (np.zeros((2, 2, 3), np.float32), np.eye(3), 1, "camera must"),
        (np.zeros((2, 2, 3), np.uint16), np.eye(2), 1, "finite 3x3"),
        (np.zeros((2, 2, 3), np.uint16), np.full((3, 3), np.nan), 1, "finite 3x3"),
        (np.zeros((2, 2, 3), np.uint16), np.eye(3), 0, "positive integer"),
    ],
)
def test_camera_raster_conversion_rejects_invalid_inputs(
    camera: np.ndarray, matrix: np.ndarray, row_block: int, match: str
) -> None:
    with pytest.raises(DngForwardRasterError, match=match):
        _apply_camera_to_rec2020(camera, matrix, row_block=row_block)


def test_public_loader_rejects_non_dng(tmp_path) -> None:
    source = tmp_path / "input.tif"
    source.write_bytes(b"not a dng")
    with pytest.raises(DngForwardRasterError, match="existing \\.dng"):
        load_dng_forward_working_image(source)
