from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.eval.filmmatch_registered_case_oracle import (
    SCHEMA,
    _operator_groups,
    register_validation_pair,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (
            ROOT
            / "configs/u5_r2be0_filmmatch_registered_case_oracle_v1.json"
        ).read_text(encoding="utf-8")
    )


def test_contract_freezes_target_fit_and_selection_closed() -> None:
    config = _config()
    assert config["schema"] == SCHEMA
    assert config["status"] == "contract_frozen_before_formal_execution"
    assert config["data"]["target_fit_allowed"] is False
    assert config["data"]["target_operator_selection_allowed"] is False
    assert config["branches"]["oracle_pass"].startswith(
        "freeze a separate target-blind"
    )


def test_operator_groups_cover_global_regimes_and_all_evs() -> None:
    records = [
        {"exposure_ev": exposure}
        for exposure in range(-5, 6)
        for _ in range(12)
    ]
    groups = _operator_groups(records)
    assert len(groups) == 15
    assert np.count_nonzero(groups["global"]) == 132
    assert np.count_nonzero(groups["regime_negative"]) == 60
    assert np.count_nonzero(groups["regime_zero"]) == 12
    assert np.count_nonzero(groups["regime_positive"]) == 60
    assert all(
        np.count_nonzero(groups[f"ev_{exposure:+d}"]) == 12
        for exposure in range(-5, 6)
    )


def test_registration_recovers_synthetic_perspective_geometry() -> None:
    rng = np.random.default_rng(20060730)
    source = np.zeros((320, 420, 3), dtype=np.uint16)
    for _ in range(240):
        x = int(rng.integers(8, 412))
        y = int(rng.integers(8, 312))
        radius = int(rng.integers(2, 6))
        color = tuple(int(value) for value in rng.integers(2000, 63000, 3))
        cv2.circle(source, (x, y), radius, color, -1)
    transform = np.asarray(
        [
            [0.94, -0.02, 18.0],
            [0.01, 0.95, 11.0],
            [0.00003, -0.00002, 1.0],
        ],
        dtype=np.float64,
    )
    target = cv2.warpPerspective(
        source, transform, (source.shape[1], source.shape[0])
    )
    contract = dict(_config()["registration"])
    contract.update(
        {
            "minimum_good_matches": 40,
            "minimum_inlier_fraction": 0.7,
            "maximum_median_inlier_reprojection_error_pixels": 1.5,
            "maximum_p95_inlier_reprojection_error_pixels": 3.0,
            "minimum_evaluation_pixel_fraction": 0.05,
            "target_code_minimum": -1.0,
            "target_code_maximum": 2.0,
            "maximum_target_luma_gradient_quantile": 1.0,
            "mask_erosion_pixels": 3,
        }
    )
    observed, mask, diagnostics = register_validation_pair(
        source, target, contract
    )
    normalized_observed = observed / observed[2, 2]
    normalized_expected = transform / transform[2, 2]
    assert diagnostics["registration_gate_passed"] is True
    assert diagnostics["inliers"] >= 40
    assert np.max(np.abs(normalized_observed - normalized_expected)) < 1.0
    assert np.mean(mask) >= 0.05
