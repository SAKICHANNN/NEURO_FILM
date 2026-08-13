from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.characteristic_scanner_chain_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4ig_characteristic_scanner_chain_d0_v1.json"


def test_characteristic_scanner_chain_passes_and_direct_control_fails() -> None:
    report = evaluate(load_contract(CONFIG), ROOT)
    assert report["automatic_pass"] is True
    assert report["metrics"]["chain_max_abs_error"] <= 4e-6
    assert report["metrics"]["direct_transmittance_control_p95_abs_error"] >= 0.25


def test_characteristic_scanner_chain_replays_exactly() -> None:
    contract = load_contract(CONFIG)
    assert evaluate(contract, ROOT) == evaluate(contract, ROOT)


def test_characteristic_scanner_chain_parent_drift_fails_closed() -> None:
    contract = deepcopy(load_contract(CONFIG))
    contract["parents"]["characteristic_trace"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="integrity"):
        evaluate(contract, ROOT)
