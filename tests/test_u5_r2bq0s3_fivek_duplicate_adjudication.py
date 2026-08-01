from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq0s3_fivek_duplicate_adjudication_v1.json"


def test_bq0s3_contract_preserves_failed_parent_and_source_only_boundary() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["status"] == "contract_frozen_implementation_ready"
    assert payload["adaptive_successor_after_bq0s2_failure"] is True
    assert payload["target_pixels_allowed"] is False
    assert payload["operator_fitting_allowed"] is False
    assert payload["router_training_allowed"] is False
    assert payload["expected_candidates"] == {
        "prior_pool": 1,
        "internal_split": 1,
    }
    assert payload["protocol"]["maximum_dhash_hamming"] == 4
    assert payload["protocol"]["maximum_phash_hamming"] == 8
    assert payload["protocol"]["minimum_normalized_luma_correlation"] == 0.85
    assert payload["protocol"]["orb"]["minimum_ransac_inliers"] == 12
