from pathlib import Path

import pytest

from src.eval.presampling_reference_convergence_64x import (
    evaluate_presampling_reference_convergence_64x,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6af_presampling_reference_convergence_v1.json"
PARENT = ROOT / "configs/u6_p6ae_presampling_reference_convergence_decision_v1.json"


def test_contract_loads():
    assert load_contract(CONTRACT)["material"]["zooms"] == [16, 32, 64]


@pytest.mark.skipif(not PARENT.is_file(), reason="P6AE decision unavailable")
def test_frozen_evaluator_repeats():
    config = load_contract(CONTRACT)
    first = evaluate_presampling_reference_convergence_64x(config, ROOT)
    second = evaluate_presampling_reference_convergence_64x(config, ROOT)
    assert first == second
    assert first["metrics"]["repeat_error"] == 0
    assert first["metrics"]["regenerated_32x_error"] == 0
