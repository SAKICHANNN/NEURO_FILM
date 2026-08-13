from copy import deepcopy
from pathlib import Path

from src.eval.fujifilm_e6_gaussian_representation_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2glut1_fujifilm_e6_gaussian_representation_d0_v1.json"


def test_e6_gaussian_representation_is_exact() -> None:
    first = evaluate(load_contract(CONFIG, ROOT), ROOT)
    assert first == evaluate(load_contract(CONFIG, ROOT), ROOT)


def test_accuracy_gate_can_only_close() -> None:
    contract = deepcopy(load_contract(CONFIG, ROOT))
    contract["gates"]["maximum_gaussian_rmse"] = 0.0
    result = evaluate(contract, ROOT)
    assert result["automatic_pass"] is False
    assert result["decision"] == contract["decision_if_fail"]
