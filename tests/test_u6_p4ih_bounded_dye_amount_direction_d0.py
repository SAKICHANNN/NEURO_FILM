from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.bounded_dye_amount_direction_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4ih_bounded_dye_amount_direction_d0_v1.json"


def test_formal_shared_direction_envelope_passes() -> None:
    report = evaluate(load_contract(CONFIG), ROOT)
    assert report["automatic_pass"] is True
    assert report["metrics"]["maximum_shared_direction_error"] <= 2e-6


def test_report_replays_exactly() -> None:
    contract = load_contract(CONFIG)
    assert evaluate(contract, ROOT) == evaluate(contract, ROOT)


def test_parent_drift_fails_closed() -> None:
    contract = deepcopy(load_contract(CONFIG))
    contract["parent"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="integrity"):
        evaluate(contract, ROOT)
