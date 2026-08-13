from copy import deepcopy
from pathlib import Path

from src.eval.gaussian_lut_representation_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2glut0_gaussian_lut_representation_d0_v1.json"


def test_gaussian_lut_d0_is_exact_and_bounded() -> None:
    first = evaluate(load_contract(CONFIG))
    assert first == evaluate(load_contract(CONFIG))
    assert first["metrics"]["maximum_new_boundary_fraction"] <= 0.0005


def test_jacobian_gate_can_only_close() -> None:
    contract = deepcopy(load_contract(CONFIG))
    contract["gates"]["minimum_jacobian_determinant"] = 10.0
    result = evaluate(contract)
    assert result["automatic_pass"] is False
    assert result["decision"] == contract["decision_if_fail"]
