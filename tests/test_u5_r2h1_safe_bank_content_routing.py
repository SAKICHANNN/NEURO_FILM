from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.safe_bank_content_routing import (
    content_descriptor,
    loo_predict,
)


def _descriptor_spec() -> dict:
    return {
        "resize": [64, 64],
        "luma_histogram_bins": 16,
        "luma_quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "saturation_histogram_bins": 8,
        "spatial_grid": [4, 4],
        "include_spatial_luma": True,
        "include_spatial_chroma_magnitude": True,
        "include_global_rgb_mean_std": True,
    }


def test_descriptor_is_deterministic_and_expected_dimension(tmp_path: Path) -> None:
    image = np.zeros((80, 90, 3), dtype=np.uint8)
    image[..., 0] = np.arange(90, dtype=np.uint8)[None, :]
    image[..., 1] = 80
    path = tmp_path / "source.png"
    Image.fromarray(image, mode="RGB").save(path)
    first = content_descriptor(path, _descriptor_spec())
    second = content_descriptor(path, _descriptor_spec())
    assert np.array_equal(first, second)
    assert first.shape == (71,)


def test_loo_standardization_and_knn_are_deterministic() -> None:
    descriptors = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.9, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    assigned = np.ones(4, dtype=bool)
    router = {
        "kind": "knn",
        "k": 1,
        "distance_weighted": False,
    }
    first, first_trace = loo_predict(descriptors, labels, assigned, router)
    second, second_trace = loo_predict(descriptors, labels, assigned, router)
    assert np.array_equal(first, labels)
    assert np.array_equal(first, second)
    assert first_trace == second_trace
    for index, row in enumerate(first_trace):
        assert index not in row["selected_training_indices"]
