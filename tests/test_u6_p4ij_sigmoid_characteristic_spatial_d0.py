from copy import deepcopy
from pathlib import Path

from src.eval.sigmoid_characteristic_spatial_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4ij_sigmoid_characteristic_spatial_d0_v1.json"


def test_sigmoid_spatial_chain_passes_or_closes_exactly() -> None:
    report = evaluate(load_contract(CONFIG), ROOT)
    assert report["decision"] in {
        "retain_sigmoid_characteristic_chain_for_bound_photo_development",
        "close_sigmoid_characteristic_scanner_family",
    }
    assert report["metrics"]["maximum_partition_error"] <= 1e-6


def test_materiality_gate_can_only_close() -> None:
    contract = deepcopy(load_contract(CONFIG))
    contract["gates"]["minimum_population_p95_abs_difference"] = 1.0
    report = evaluate(contract, ROOT)
    assert report["automatic_pass"] is False
    assert report["decision"] == contract["decision_if_fail"]
