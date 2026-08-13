from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.rail_guarded_neutral_structure_ao6_d0 import (
    apply_rail_guarded_neutral_structure,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4is_rail_guarded_neutral_structure_ao6_d0_v1.json"


def test_p4is_guarded_structure_is_bounded_and_nontrivial() -> None:
    value = np.linspace(0.0, 1.0, 16 * 12 * 3, dtype=np.float32).reshape(12, 16, 3)
    result = apply_rail_guarded_neutral_structure(
        value,
        value,
        1,
        0.018,
        hard_boundary_epsilon=1.0 / 510.0,
        guard_boundary_epsilon=1.0 / 255.0,
    )
    assert result.dtype == np.float32
    assert np.min(result) >= 0.0 and np.max(result) <= 1.0
    assert np.max(np.abs(result - value)) > 1e-4


def test_p4is_formal_replay_and_decision() -> None:
    contract = load_contract(CONFIG)
    first = evaluate(contract, ROOT)
    second = evaluate(contract, ROOT)
    assert first == second
    assert first["decision"] in {contract["decision_if_pass"], contract["decision_if_fail"]}
