from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3x_anchored_bracket_fusion_recovery_v1.json"


def test_p3x_contract_freezes_physical_variance_weighted_fusion() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    fusion = contract["fusion"]
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert fusion["observations_per_group"] == 3
    assert fusion["estimator"] == "per-scalar inverse-variance weighted mean"
    assert fusion["target_reads"] is False
    assert fusion["post_score_changes_allowed"] is False


def test_p3x_contract_keeps_p3u_gates_and_hard_10bit_case() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["anchor_precision_levels_ppm"] == [0, 100]
    assert contract["required_passing_levels_ppm"] == [0, 100]
    assert contract["unchanged_evaluation"]["automatic_gates"] == (
        "exact P3U per-regime and common gates"
    )
    forbidden = " ".join(contract["forbidden"])
    assert "after scoring" in forbidden
    assert "10-bit regime" in forbidden
