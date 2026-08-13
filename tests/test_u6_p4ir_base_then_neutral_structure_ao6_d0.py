from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.base_then_neutral_structure_ao6_d0 import (
    apply_post_base_neutral_structure,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4ir_base_then_neutral_structure_ao6_d0_v1.json"


def test_p4ir_post_base_structure_is_bounded_and_nontrivial() -> None:
    value = np.linspace(0.02, 0.98, 16 * 12 * 3, dtype=np.float32).reshape(12, 16, 3)
    result = apply_post_base_neutral_structure(value, value, 1, 0.018)
    assert result.dtype == np.float32
    assert np.min(result) >= 0.0 and np.max(result) <= 1.0
    assert np.max(np.abs(result - value)) > 1e-4


def test_p4ir_formal_replay_and_decision() -> None:
    contract = load_contract(CONFIG)
    first = evaluate(contract, ROOT)
    second = evaluate(contract, ROOT)
    assert first == second
    assert first["decision"] in {contract["decision_if_pass"], contract["decision_if_fail"]}
    assert first["blind_review_allowed"] is False
