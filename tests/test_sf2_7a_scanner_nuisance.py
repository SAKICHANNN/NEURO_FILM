from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.real_film.scanner_nuisance import (
    _native_rgb,
    apply_affine,
    distance_metrics,
    evaluate_leave_one_slide_out,
    patch_medians,
)


def test_native_rgb_preserves_integer_precision() -> None:
    import tifffile

    values = np.array([[[0, 1024, 65535], [17, 32768, 60000]]], dtype=np.uint16)
    payload = BytesIO()
    tifffile.imwrite(payload, values, photometric="rgb")
    decoded = _native_rgb(payload.getvalue())
    assert decoded.dtype == np.float64
    np.testing.assert_allclose(decoded, values.astype(np.float64) / 65535.0)


def test_native_rgb_falls_back_for_missing_imagecodecs() -> None:
    values = np.array([[[0, 64, 255], [17, 128, 200]]], dtype=np.uint8)
    payload = BytesIO()
    Image.fromarray(values, mode="RGB").save(payload, format="TIFF")
    with patch(
        "src.real_film.scanner_nuisance.tifffile.imread",
        side_effect=ValueError(
            "<COMPRESSION.LZW: 5> requires the 'imagecodecs' package"
        ),
    ):
        decoded = _native_rgb(payload.getvalue())
    np.testing.assert_allclose(decoded, values.astype(np.float64) / 255.0)


def test_patch_medians_extract_fixed_grid() -> None:
    image = np.zeros((40, 60, 3), dtype=np.float64)
    image[10:30, 10:50] = [0.2, 0.4, 0.6]
    grid = {
        "columns": 2,
        "rows": 1,
        "left_edge_px": 10.0,
        "right_edge_px": 50.0,
        "top_edge_px": 10.0,
        "bottom_edge_px": 30.0,
        "inner_patch_fraction": 0.5,
    }
    medians = patch_medians(image, np.eye(3), grid)
    assert medians.shape == (2, 3)
    np.testing.assert_allclose(medians, [[0.2, 0.4, 0.6]] * 2)


def test_affine_application_is_bounded() -> None:
    source = np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])
    matrix = np.eye(3) * 2.0
    bias = np.array([-0.25, 0.0, 0.25])
    output = apply_affine(source, matrix, bias)
    assert np.min(output) == 0.0
    assert np.max(output) == 1.0


def test_distance_metrics_are_exact_for_identity() -> None:
    values = np.array([[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]])
    metrics = distance_metrics(values, values)
    assert metrics["median_rgb_euclidean"] == 0.0
    assert metrics["median_delta_e00_srgb_assumption"] == 0.0


def test_leave_one_slide_out_recovers_global_affine() -> None:
    roles = ["a", "b", "c", "d"]
    slides = ["1", "2", "3", "4", "5"]
    rng = np.random.default_rng(7)
    base = {slide: rng.uniform(0.1, 0.7, size=(8, 3)) for slide in slides}
    scales = {
        "a": np.array([1.0, 1.0, 1.0]),
        "b": np.array([0.8, 1.1, 1.2]),
        "c": np.array([1.2, 0.9, 0.7]),
        "d": np.array([0.7, 0.8, 1.1]),
    }
    patch_bank = {
        role: {
            slide: np.clip(base[slide] * scales[role], 0.0, 1.0)
            for slide in slides
        }
        for role in roles
    }
    config = {
        "pipeline_roles": roles,
        "slide_ids": slides,
        "evaluation": {
            "models": [
                "identity",
                "per_channel_affine",
                "bounded_full_affine_3x3_plus_bias",
            ],
            "per_channel_scale_bounds": [0.0, 2.0],
            "per_channel_bias_bounds": [-0.5, 0.5],
            "full_affine_coefficient_bounds": [-2.0, 2.0],
            "full_affine_bias_bounds": [-0.5, 0.5],
        },
        "pre_registered_interpretation": {
            "material_raw_nuisance_aggregate_median_rgb_minimum": 0.01,
            "material_raw_nuisance_each_pair_median_rgb_minimum": 0.005,
            "canonicalizer_material_reduction_fraction_minimum": 0.5,
            "small_residual_aggregate_median_rgb_maximum": 1e-6,
        },
    }
    result = evaluate_leave_one_slide_out(patch_bank, config)
    assert result["aggregate"]["identity"]["median_rgb_euclidean"] > 0.01
    assert result["aggregate"]["per_channel_affine"]["median_rgb_euclidean"] < 1e-8
    assert result["interpretation"]["canonicalizer_material_reduction"]
    assert result["interpretation"]["small_residual"]
