from copy import deepcopy
from pathlib import Path

from src.eval.characteristic_scanner_spatial_smoke import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4ii_characteristic_scanner_spatial_smoke_v1.json"


def test_spatial_smoke_is_exact_safe_and_material() -> None:
    report = evaluate(load_contract(CONFIG), ROOT)
    assert report["automatic_pass"] is True
    assert report["metrics"]["maximum_partition_error"] <= 1e-6


def test_gate_failure_is_scientific_not_contract_rescue() -> None:
    contract = deepcopy(load_contract(CONFIG))
    contract["gates"]["minimum_population_p95_abs_difference"] = 1.0
    report = evaluate(contract, ROOT)
    assert report["automatic_pass"] is False
    assert report["decision"] == contract["decision_if_fail"]
