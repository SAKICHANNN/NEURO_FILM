from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.preprocess.ocio_aces2_output import (
    OcioAces2RuntimeError,
    apply_working_image_aces2_output,
    convert_working_image_to_acescg,
)
from src.preprocess.types import SourceProfile, WorkingImage


def _working(
    pixels: np.ndarray,
    *,
    working_space: str = "linear_srgb",
    transfer_state: str = "scene_linear",
) -> WorkingImage:
    return WorkingImage(
        pixels=pixels,
        working_space=working_space,
        transfer_state=transfer_state,
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "test"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path("fixture.dng"),
    )


@pytest.mark.parametrize("working_space", ["linear_srgb", "linear_rec2020"])
@pytest.mark.parametrize("target", ["sdr_rec709", "hdr_rec2020_pq"])
def test_working_image_adapter_is_finite_nonmutating_and_contiguous(
    working_space: str, target: str
) -> None:
    pixels = np.asarray(
        [[[-0.01, 0.0, 0.18], [1.0, 4.0, 0.01]]], dtype=np.float32
    )
    original = pixels.copy()
    output = apply_working_image_aces2_output(
        _working(pixels, working_space=working_space), target
    )
    np.testing.assert_array_equal(pixels, original)
    assert output.shape == pixels.shape
    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    assert np.isfinite(output).all()


def test_conversion_to_acescg_changes_space_without_mutating_input() -> None:
    pixels = np.asarray([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    converted = convert_working_image_to_acescg(_working(pixels))
    assert converted.shape == pixels.shape
    assert not np.array_equal(converted, pixels)
    np.testing.assert_array_equal(pixels, np.asarray([[[0.1, 0.2, 0.3]]], np.float32))


@pytest.mark.parametrize(
    ("working_space", "transfer_state", "message"),
    [
        ("acescg", "scene_linear", "linear_srgb or linear_rec2020"),
        ("linear_srgb", "display_linear", "scene_linear"),
        ("linear_rec2020", "display_referred", "scene_linear"),
    ],
)
def test_working_image_adapter_rejects_unsupported_boundaries(
    working_space: str, transfer_state: str, message: str
) -> None:
    with pytest.raises(OcioAces2RuntimeError, match=message):
        convert_working_image_to_acescg(
            _working(
                np.zeros((1, 1, 3), dtype=np.float32),
                working_space=working_space,
                transfer_state=transfer_state,
            )
        )


def test_working_image_adapter_rejects_non_working_image() -> None:
    with pytest.raises(OcioAces2RuntimeError, match="WorkingImage"):
        convert_working_image_to_acescg(object())  # type: ignore[arg-type]
