from copy import deepcopy
from pathlib import Path

from src.eval.statistical_lut_identifiability_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2statlut0_statistical_lut_identifiability_d0_v1.json"


def test_protocol_is_exact_and_spatially_agnostic() -> None:
    first = evaluate(load_contract(CONFIG))
    second = evaluate(load_contract(CONFIG))
    assert first == second
    assert first["metrics"]["patch_shuffle_descriptor_max_error"] <= 1e-12
    assert first["metrics"]["maximum_new_boundary_fraction"] == 0.0


def test_frozen_accuracy_gate_can_only_close() -> None:
    contract = deepcopy(load_contract(CONFIG))
    contract["gates"]["minimum_top1_accuracy"] = 1.01
    result = evaluate(contract)
    assert result["automatic_pass"] is False
    assert result["decision"] == contract["decision_if_fail"]
