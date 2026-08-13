from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.negative_scanner_inverse_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4if_negative_scanner_inverse_d0_v2.json"


def test_formal_inverse_passes_and_endpoint_control_fails() -> None:
    report = evaluate(load_contract(CONFIG), ROOT)
    assert report["automatic_pass"] is True
    assert report["metrics"]["inverse_density_max_abs"] <= 4e-6
    assert report["metrics"]["endpoint_control_p95_abs"] >= 0.25


def test_report_replays_exactly() -> None:
    config = load_contract(CONFIG)
    assert evaluate(config, ROOT) == evaluate(config, ROOT)


def test_parent_drift_fails_closed() -> None:
    config = deepcopy(load_contract(CONFIG))
    config["parent"]["evidence"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="integrity"):
        evaluate(config, ROOT)
