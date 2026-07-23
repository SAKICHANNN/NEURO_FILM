from __future__ import annotations

import numpy as np
import pytest
from skimage.color import rgb2lab

from src.roll2film.lab_statistics import (
    LAB_STATISTICS_SCHEMA,
    extract_lab_statistics,
)
from src.roll2film.lut import DenseLUT3D


def _identity_lut(size: int = 2) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def test_descriptor_shapes_mass_schema_and_immutability() -> None:
    image = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    descriptor = extract_lab_statistics(image)

    assert descriptor.lightness_histogram.shape == (256,)
    assert descriptor.chroma_histogram_sqrt.shape == (32, 32)
    assert descriptor.chroma_conditioned_mean_lightness.shape == (32, 32)
    assert descriptor.raw_chroma_histogram.shape == (32, 32)
    assert descriptor.vector().shape == (2304,)
    assert np.isclose(descriptor.lightness_histogram.sum(), 1.0, atol=1e-12)
    assert np.isclose(descriptor.raw_chroma_histogram.sum(), 1.0, atol=1e-12)
    assert np.all(np.isfinite(descriptor.vector()))
    assert descriptor.to_dict()["schema"] == LAB_STATISTICS_SCHEMA
    with pytest.raises(ValueError):
        descriptor.lightness_histogram[0] = 1.0


def test_lightness_soft_bins_match_independent_two_pixel_reference() -> None:
    image = np.array([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]])
    descriptor = extract_lab_statistics(image)
    lightness = rgb2lab(image).reshape(-1, 3)[:, 0]
    expected = np.zeros(256)
    for value in lightness:
        coordinate = np.clip(value, 0.0, 100.0) * 255.0 / 100.0
        lower = int(np.floor(coordinate))
        upper = min(lower + 1, 255)
        expected[lower] += 1.0 - (coordinate - lower)
        expected[upper] += coordinate - lower
    expected /= 2.0
    assert np.allclose(descriptor.lightness_histogram, expected, atol=1e-14)


def test_permutation_invariance_and_determinism() -> None:
    rng = np.random.default_rng(20260723)
    pixels = rng.random((257, 3))
    permuted = pixels[rng.permutation(len(pixels))]
    first = extract_lab_statistics(pixels)
    second = extract_lab_statistics(permuted)
    repeated = extract_lab_statistics(pixels)

    assert np.max(np.abs(first.vector() - second.vector())) <= 1e-12
    assert np.array_equal(first.vector(), repeated.vector())


def test_palette_dependence_witness() -> None:
    red = np.tile(np.array([[0.85, 0.10, 0.08]]), (256, 1))
    blue = np.tile(np.array([[0.06, 0.12, 0.90]]), (256, 1))
    distance = np.linalg.norm(
        extract_lab_statistics(red).vector() - extract_lab_statistics(blue).vector()
    )
    assert distance > 0.1


def test_absent_colour_two_lut_non_identifiability_witness() -> None:
    identity_values = _identity_lut()
    changed_values = identity_values.copy()
    changed_values[0, 0, 1] = np.array([1.0, 0.0, 1.0])
    identity = DenseLUT3D(identity_values, np.zeros(3), np.ones(3), "tetrahedral")
    changed = DenseLUT3D(changed_values, np.zeros(3), np.ones(3), "tetrahedral")
    reference = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    probe = np.array([[[0.0, 0.0, 1.0]]])

    reference_identity = identity.apply(reference)
    reference_changed = changed.apply(reference)
    descriptor_identity = extract_lab_statistics(reference_identity).vector()
    descriptor_changed = extract_lab_statistics(reference_changed).vector()
    probe_difference = np.linalg.norm(identity.apply(probe) - changed.apply(probe))

    assert np.max(np.abs(reference_identity - reference_changed)) <= 1e-12
    assert np.max(np.abs(descriptor_identity - descriptor_changed)) <= 1e-12
    assert probe_difference >= 0.1


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((4, 4)),
        np.zeros((0, 3)),
        np.array([[[-0.1, 0.0, 0.0]]]),
        np.array([[[1.1, 0.0, 0.0]]]),
        np.array([[[np.nan, 0.0, 0.0]]]),
    ],
)
def test_invalid_inputs_fail_closed(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        extract_lab_statistics(invalid)
