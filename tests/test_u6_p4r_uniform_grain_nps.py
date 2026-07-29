from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from src.eval.real_uniform_grain_nps import (
    UniformGrainNpsError,
    evaluate_signatures,
    exact_balanced_label_permutation,
    fixed_fractional_crops,
    radial_nps_signature,
)


EDGES = np.geomspace(1.0 / 64.0, 0.45, 17).tolist()


def _noise(seed: int, sigma: float) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(size=(256, 256))
    return 0.5 + 0.03 * gaussian_filter(noise, sigma=sigma)


def test_signature_ignores_constant_scale_and_quadratic_shading() -> None:
    base = _noise(4, 1.2)
    y, x = np.mgrid[-1.0:1.0:256j, -1.0:1.0:256j]
    transformed = 1.8 * base + 0.1 + 0.03 * x - 0.02 * y + 0.01 * x * y
    left = radial_nps_signature(base, band_edges_cycles_per_pixel=EDGES)
    right = radial_nps_signature(
        transformed,
        band_edges_cycles_per_pixel=EDGES,
    )
    assert float(np.dot(left, right)) > 0.999


def test_fixed_fractional_crops_do_not_select_by_content() -> None:
    image = np.arange(512 * 768, dtype=np.float32).reshape(512, 768)
    crops = fixed_fractional_crops(
        image,
        crop_size=128,
        centers_yx=[[0.25, 0.25], [0.75, 0.75]],
    )
    assert [crop.shape for crop in crops] == [(128, 128), (128, 128)]
    with pytest.raises(UniformGrainNpsError, match="leaves image"):
        fixed_fractional_crops(
            image,
            crop_size=400,
            centers_yx=[[0.1, 0.1]],
        )


def test_exact_scan_group_classifier_separates_two_nps_shapes() -> None:
    signatures = []
    labels = []
    crops = []
    for stock_index, sigma in enumerate((1.0, 3.0)):
        for scan in range(4):
            scan_rows = []
            for crop in range(4):
                signature = radial_nps_signature(
                    _noise(100 * stock_index + 10 * scan + crop, sigma),
                    band_edges_cycles_per_pixel=EDGES,
                )
                scan_rows.append(signature)
            signatures.append(np.mean(scan_rows, axis=0))
            labels.append(f"stock-{stock_index}")
            crops.append(np.asarray(scan_rows))
    classifier = exact_balanced_label_permutation(
        np.asarray(signatures),
        labels,
    )
    assert classifier["balanced_accuracy"] == 1.0
    result = evaluate_signatures(
        scan_ids=[f"scan-{index}" for index in range(8)],
        stock_ids=labels,
        crop_signatures=crops,
        gates={
            "minimum_median_crop_to_scan_similarity": 0.85,
            "minimum_median_within_stock_similarity": 0.9,
            "minimum_leave_one_scan_out_balanced_accuracy": 0.75,
            "maximum_exact_permutation_p_value": 0.1,
            "minimum_within_minus_cross_similarity": 0.03,
        },
    )
    assert result["repeatability_pass"] is True
    assert result["stock_association_pass"] is True


def test_association_failure_retains_only_generic_repeatable_shape() -> None:
    shared = np.tile(np.asarray([1.0, -1.0]) / np.sqrt(2.0), (8, 4, 1))
    result = evaluate_signatures(
        scan_ids=[f"scan-{index}" for index in range(8)],
        stock_ids=["a"] * 4 + ["b"] * 4,
        crop_signatures=[row for row in shared],
        gates={
            "minimum_median_crop_to_scan_similarity": 0.85,
            "minimum_median_within_stock_similarity": 0.9,
            "minimum_leave_one_scan_out_balanced_accuracy": 0.75,
            "maximum_exact_permutation_p_value": 0.1,
            "minimum_within_minus_cross_similarity": 0.03,
        },
    )
    assert result["repeatability_pass"] is True
    assert result["stock_association_pass"] is False
    assert result["branch"] == "retain_generic_scanner_convolved_signature_only"
