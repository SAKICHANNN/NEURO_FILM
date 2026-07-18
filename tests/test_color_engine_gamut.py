from __future__ import annotations

import numpy as np
import pytest

from src.color_engine import (
    compress_chroma_to_working_gamut,
    compress_source_to_working_gamut,
    in_working_gamut,
    linear_rgb_to_lab,
)


def _source_lab() -> np.ndarray:
    pixels = np.random.default_rng(14036).uniform(0.08, 0.92, size=(9, 11, 3)).astype(np.float32)
    return linear_rgb_to_lab(pixels, working_space="linear_rec2020")


def test_source_compression_returns_destination_gamut_without_mutation() -> None:
    source = _source_lab()
    frozen_source = source.copy()
    target = source.copy()
    target[..., 1] += 180.0
    target[..., 2] -= 140.0
    frozen_target = target.copy()

    output = compress_source_to_working_gamut(
        source,
        target,
        working_space="linear_rec2020",
    )

    assert in_working_gamut(output, working_space="linear_rec2020", tolerance=2e-6).all()
    np.testing.assert_array_equal(source, frozen_source)
    np.testing.assert_array_equal(target, frozen_target)


def test_chroma_compression_preserves_luminance_and_hue_direction() -> None:
    target = _source_lab()
    target[..., 1:] *= 7.0
    output = compress_chroma_to_working_gamut(
        target,
        working_space="linear_rec2020",
    )

    assert in_working_gamut(output, working_space="linear_rec2020", tolerance=2e-6).all()
    np.testing.assert_array_equal(output[..., 0], target[..., 0])
    cross = output[..., 1] * target[..., 2] - output[..., 2] * target[..., 1]
    norm_product = np.maximum(
        np.linalg.norm(output[..., 1:], axis=-1) * np.linalg.norm(target[..., 1:], axis=-1),
        1e-6,
    )
    assert float(np.max(np.abs(cross) / norm_product)) <= 2e-6
    assert np.all(np.sum(output[..., 1:] * target[..., 1:], axis=-1) >= 0.0)


def test_source_compression_rejects_out_of_gamut_source_endpoint() -> None:
    source = _source_lab()
    source[..., 1] += 300.0
    with pytest.raises(ValueError, match="source Lab endpoint"):
        compress_source_to_working_gamut(
            source,
            source.copy(),
            working_space="linear_rec2020",
        )


def test_chroma_compression_rejects_out_of_gamut_neutral_endpoint() -> None:
    target = np.full((3, 4, 3), [140.0, 60.0, -40.0], dtype=np.float32)
    with pytest.raises(ValueError, match="neutral endpoint"):
        compress_chroma_to_working_gamut(
            target,
            working_space="linear_rec2020",
        )


@pytest.mark.parametrize("iterations", [0, -1])
def test_gamut_compression_rejects_nonpositive_iterations(iterations: int) -> None:
    source = _source_lab()
    with pytest.raises(ValueError, match="iterations"):
        compress_source_to_working_gamut(
            source,
            source,
            working_space="linear_rec2020",
            iterations=iterations,
        )
