from copy import deepcopy
from pathlib import Path

from src.eval.curvature_representation_selector_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2glut2_curvature_representation_selector_d0_v1.json"


def test_selector_is_exact() -> None:
    first = evaluate(load_contract(CONFIG), ROOT)
    assert first == evaluate(load_contract(CONFIG), ROOT)


def test_accuracy_gate_can_only_close() -> None:
    contract = deepcopy(load_contract(CONFIG))
    contract["gates"]["minimum_selector_accuracy"] = 1.01
    result = evaluate(contract, ROOT)
    assert result["automatic_pass"] is False
    assert result["decision"] == contract["decision_if_fail"]
