from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3w_external_exposure_anchor_sufficiency_v1.json"


def test_p3w_contract_freezes_anchor_ladder_before_scoring() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    levels = contract["anchor_precision_levels_ppm"]
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert levels == [0, 100, 250, 500, 1000]
    assert levels == sorted(levels)
    assert contract["required_passing_levels_ppm"] == [0, 100]


def test_p3w_contract_keeps_anchor_independent_and_p3u_gates_unchanged() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["roles"]["anchor_reads_targets"] is False
    assert contract["roles"]["mechanism_fit_reads_confirmation"] is False
    assert "targets" in contract["anchor_observation"]["forbidden_inputs"]
    assert contract["unchanged_evaluation"]["automatic_gates"] == (
        "exact P3U per-regime and common gates"
    )
    forbidden = " ".join(contract["forbidden"])
    assert "after scoring" in forbidden
    assert "favorable error draws" in forbidden
