from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.rotated_plate_directional_attribution_d0 import (
    _classify_profiles,
    _rank_center_crop,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_contract_forbids_registration_and_postscore_rescue() -> None:
    contract = load_contract(
        ROOT / "configs/u6_p6ax_rotated_plate_directional_attribution_d0_v1.json"
    )
    assert contract["analysis"]["registration_allowed"] is False
    assert contract["gates"]["required_plates"] == 2
    assert "threshold" in contract["forbidden"][0]


def test_rank_crop_uses_full_image_distribution() -> None:
    values = np.arange(36, dtype=np.uint16).reshape(6, 6)
    result = _rank_center_crop(values, 2)
    expected = (np.array([[14, 15], [20, 21]]) + 0.5) / 36
    assert np.array_equal(result, expected)


def test_classifier_distinguishes_plate_following_rotation() -> None:
    reference_a = np.zeros((32, 32), dtype=np.float64)
    reference_a[8:24, 14:18] = 1.0
    reference_b = np.zeros((32, 32), dtype=np.float64)
    reference_b[5:9, 4:27] = 1.0
    mask = np.ones((32, 32), dtype=bool)
    rows = _classify_profiles(
        [
            {"reference": reference_a, "rotated": np.rot90(reference_a, -1), "mask": mask},
            {"reference": reference_b, "rotated": np.rot90(reference_b, -1), "mask": mask},
        ],
        0.05,
    )
    assert all(row["plate_following_pass"] for row in rows)
    assert not any(row["scanner_fixed_pass"] for row in rows)


def test_classifier_distinguishes_scanner_fixed_structure() -> None:
    reference_a = np.zeros((32, 32), dtype=np.float64)
    reference_a[:, 7:9] = 1.0
    reference_b = np.zeros((32, 32), dtype=np.float64)
    reference_b[::3, :] = 1.0
    mask = np.ones((32, 32), dtype=bool)
    rows = _classify_profiles(
        [
            {"reference": reference_a, "rotated": reference_a.copy(), "mask": mask},
            {"reference": reference_b, "rotated": reference_b.copy(), "mask": mask},
        ],
        0.05,
    )
    assert all(row["scanner_fixed_pass"] for row in rows)
    assert not any(row["plate_following_pass"] for row in rows)
