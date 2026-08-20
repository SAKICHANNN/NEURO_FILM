from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.barnard_fullscan_directional_confirmation import _score, load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_contract_reuses_exact_p6ax_analysis_and_keeps_claim_narrow() -> None:
    p6ax = __import__("json").loads(
        (ROOT / "configs/u6_p6ax_rotated_plate_directional_attribution_d0_v1.json").read_text(
            encoding="utf-8"
        )
    )
    contract = load_contract(
        ROOT / "configs/u6_p6ay_barnard_fullscan_directional_confirmation_d0_v1.json"
    )
    assert contract["analysis"] == p6ax["analysis"]
    assert "never a third plate" in (
        ROOT / "configs/u6_p6ay_barnard_fullscan_directional_confirmation_source_v1.json"
    ).read_text(encoding="utf-8")
    assert "population" in contract["claim_ceiling"]


def test_score_retains_plate_rotation_and_rejects_controls() -> None:
    target = np.zeros((32, 32), dtype=np.float64)
    target[6:25, 13:17] = 1.0
    wrong = np.zeros((32, 32), dtype=np.float64)
    wrong[4:8, 5:27] = 1.0
    mask = np.ones((32, 32), dtype=bool)
    metrics = _score(target, np.rot90(target, -1), wrong, wrong, mask)
    assert metrics["plate_following_correlation"] == 1.0
    assert metrics["plate_following_margin"] > 0.5
    assert metrics["scanner_fixed_margin"] < 0.0
