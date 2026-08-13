from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.typed_log_scanner_conformance import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4ie_typed_log_scanner_conformance_v1.json"


def test_formal_conformance_passes_and_replays_exactly() -> None:
    config = load_contract(CONFIG)
    first = evaluate(config, ROOT)
    second = evaluate(config, ROOT)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["maximum_float32_vs_float64_absolute_error"] <= 2e-7
    assert first["maximum_repeat_absolute_error"] == 0.0


def test_parent_hash_drift_fails_closed() -> None:
    config = deepcopy(load_contract(CONFIG))
    config["parent"]["implementation"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="integrity"):
        evaluate(config, ROOT)


def test_positive_density_slope_is_rejected() -> None:
    config = deepcopy(load_contract(CONFIG))
    config["compiler"]["matrix_density_to_log10_rgb"][0][0] = 0.01
    with pytest.raises(ValueError, match="nonincreasing"):
        evaluate(config, ROOT)
