from __future__ import annotations

import numpy as np
import pytest
from skimage.color import rgb2lab

from src.color_engine import lab_to_linear_rgb, linear_rgb_to_lab
from src.preprocess import convert_linear_rgb


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def test_linear_srgb_lab_matches_frozen_skimage_convention() -> None:
    encoded = np.random.default_rng(14031).random((17, 19, 3), dtype=np.float32)
    actual = linear_rgb_to_lab(_srgb_to_linear(encoded), working_space="linear_srgb")
    expected = rgb2lab(encoded).astype(np.float32)
    assert float(np.max(np.abs(actual - expected))) <= 3e-4


@pytest.mark.parametrize("working_space", ["linear_srgb", "linear_rec2020"])
def test_d65_lab_roundtrip_preserves_extended_linear_values(working_space: str) -> None:
    pixels = np.random.default_rng(14032).uniform(
        -0.05, 1.20, size=(13, 11, 3)
    ).astype(np.float32)
    lab = linear_rgb_to_lab(pixels, working_space=working_space)
    restored = lab_to_linear_rgb(lab, working_space=working_space)
    assert float(np.max(np.abs(restored - pixels))) <= 3e-6


def test_rec2020_primaries_are_finite_and_not_srgb_interpreted() -> None:
    primaries = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
    rec2020_lab = linear_rgb_to_lab(primaries, working_space="linear_rec2020")
    srgb_lab = linear_rgb_to_lab(primaries, working_space="linear_srgb")
    assert np.isfinite(rec2020_lab).all()
    assert float(np.max(np.abs(rec2020_lab - srgb_lab))) > 10.0


def test_same_physical_colour_has_stable_lab_across_working_spaces() -> None:
    linear_srgb = np.random.default_rng(14033).random((13, 17, 3), dtype=np.float32)
    linear_rec2020 = convert_linear_rgb(
        linear_srgb,
        source_space="linear_srgb",
        destination_space="linear_rec2020",
    )
    srgb_lab = linear_rgb_to_lab(linear_srgb, working_space="linear_srgb")
    rec2020_lab = linear_rgb_to_lab(
        linear_rec2020, working_space="linear_rec2020"
    )
    assert float(np.max(np.abs(srgb_lab - rec2020_lab))) <= 1e-4


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((3, 4), dtype=np.float32),
        np.zeros((3, 4, 3), dtype=np.float64),
        np.full((3, 4, 3), np.nan, dtype=np.float32),
    ],
)
def test_lab_boundary_rejects_invalid_arrays(value: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        linear_rgb_to_lab(value, working_space="linear_rec2020")
    with pytest.raises((TypeError, ValueError)):
        lab_to_linear_rgb(value, working_space="linear_rec2020")
