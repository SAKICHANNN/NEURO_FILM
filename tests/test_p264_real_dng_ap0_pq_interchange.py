from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.audit_p264_real_dng_ap0_pq_interchange import _compare_samples

ROOT = Path(__file__).resolve().parents[1]


def test_p264_contract_freezes_one_existing_real_dng() -> None:
    config = json.loads(
        (ROOT / "configs/p264_real_dng_ap0_pq_interchange_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["status"] == "FROZEN_BEFORE_IMPLEMENTATION_OR_OUTPUT_COMPARISON"
    assert config["input"]["source_id"] == "blackmagic_pocket_cinema_camera_4k"
    assert config["input"]["shape"] == [2176, 4128, 3]
    assert config["gates"]["maximum_rgb16_code_difference"] == 0
    assert "candidate 3" in config["claim_ceiling"]


def test_p264_sample_comparison_is_exact_and_discriminating() -> None:
    direct = np.array([[[0, 1, 65535], [9, 10, 11]]], dtype=np.uint16)
    exact = _compare_samples(direct, direct.copy())
    assert exact == {
        "different_component_count": 0,
        "maximum_code_difference": 0,
        "rgb16_bytes_exact": True,
    }
    changed = direct.copy()
    changed[0, 1, 2] += 1
    comparison = _compare_samples(direct, changed)
    assert comparison["different_component_count"] == 1
    assert comparison["maximum_code_difference"] == 1
    assert comparison["rgb16_bytes_exact"] is False
