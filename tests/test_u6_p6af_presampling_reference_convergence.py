from pathlib import Path

import pytest

from src.eval.presampling_reference_convergence_64x import (
    evaluate_presampling_reference_convergence_64x,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6af_presampling_reference_convergence_v1.json"
PARENT = ROOT / "configs/u6_p6ae_presampling_reference_convergence_decision_v1.json"
FORMAL_REPORT = (
    ROOT
    / "outputs/experiments/u6_p6af_presampling_reference_convergence_v1/report_run1.json"
)


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


@pytest.mark.skipif(not FORMAL_REPORT.is_file(), reason="P6AF report unavailable")
def test_formal_result_is_stably_bound():
    import hashlib
    import json

    payload = FORMAL_REPORT.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        "6a04878276c1437df3132265ce5412e71b08f1d4da07bf8c0fe34fe07f27397b"
    )
    assert report["stable_evidence_id"] == (
        "6134aed4341dbb938cd9596b1f201e7435332f21f55b73bf38cded31b1bcc6e2"
    )
    assert report["automatic_pass"] is False
    assert report["decision"] == "close_raster_presampling_reference_family"
