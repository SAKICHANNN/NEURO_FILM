from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.real_film.three_stock_capture_alignment import (
    load_contract,
    stress_image,
    truth_reprojection_errors,
)

ROOT = Path(__file__).resolve().parents[1]


def test_sf3_a0m_contract_keeps_stock_fitting_closed() -> None:
    contract = load_contract(ROOT / "configs/sf3_a0m_three_stock_capture_alignment_stress_v1.json")
    assert contract["evaluation"]["pixel_operator_fit_allowed"] is False
    assert contract["evaluation"]["stock_operator_fit_allowed"] is False
    assert contract["gates"]["required_rows"] == 36


def test_sf3_a0m_truth_metric_recovers_exact_homography() -> None:
    yy, xx = np.indices((192, 256))
    source = np.stack(
        [((xx * 17 + yy * 3) % 251), ((xx * 5 + yy * 19) % 253), ((xx ^ yy) % 255)],
        axis=2,
    ).astype(np.float32) / 255.0
    condition = load_contract(ROOT / "configs/sf3_a0m_three_stock_capture_alignment_stress_v1.json")["stress_conditions"][0]
    stressed, truth = stress_image(source, condition)
    assert stressed.shape == source.shape
    errors = truth_reprojection_errors(
        truth,
        truth,
        source.shape[:2],
        {"inner_margin_fraction": 0.05, "truth_grid_columns": 9, "truth_grid_rows": 7},
    )
    assert float(np.max(errors)) == 0.0
    assert cv2.determinant(truth) != 0.0
