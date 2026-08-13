from __future__ import annotations

from pathlib import Path

from src.eval.ao6_neutral_structure_stage_attribution import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4iq_ao6_stage_attribution_v1.json"


def test_p4iq_formal_replay_and_complete_stage_attribution() -> None:
    contract = load_contract(CONFIG)
    first = evaluate(contract, ROOT)
    second = evaluate(contract, ROOT)
    assert first == second
    assert set(first["aggregate"]) == set(contract["stages"])
    assert len(first["rows"]) == 5
