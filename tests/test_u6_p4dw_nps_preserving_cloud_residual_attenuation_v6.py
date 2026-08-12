from __future__ import annotations

import json
from pathlib import Path

from src.eval.nps_preserving_cloud_residual_attenuation_v6 import evaluate

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p4dw_nps_preserving_cloud_residual_attenuation_v6.json"
)


def test_p4dw_contract_uses_fresh_roles_and_attenuation() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["fixture"]["development_seeds"] == [57203, 57221, 57223]
    assert contract["fixture"]["confirmation_seeds"] == [54217, 54227, 54251]
    assert contract["gates"]["minimum_gain"] == 0.5
    assert contract["gates"]["maximum_gain"] == 1.0


def test_p4dw_evaluation_respects_frozen_gates() -> None:
    report = evaluate(ROOT, CONTRACT)
    gain = report["stable"]["compiled_channel_gain"]
    assert all(0.5 <= value <= 1.0 for value in gain)
    assert report["stable"]["gates"]["repeat"] is True
