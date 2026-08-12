from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.scanner_unmixing_layer_correlation import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cr_scanner_unmixing_layer_correlation_v1.json"


def test_p4cr_contract_binds_closed_archival_proxy() -> None:
    contract = load_contract(ROOT, CONTRACT)
    assert contract["experiment_id"] == "u6.p4cr-scanner-unmixing-layer-correlation-v1"
    assert contract["simulation"]["development_seeds"] != contract["simulation"]["confirmation_seeds"]
    assert contract["metrics"]["maximum_correct_inverse_rmse"] == 0.01


def test_p4cr_correct_physical_inverse_recovers_layer_correlation() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is True
    errors = report["stable"]["confirmation_median_errors"]
    assert errors["correct_inverse_density"] < errors["scanner_rgb"]
    assert errors["correct_inverse_density"] < errors["wrong_inverse_density"]
    assert all(report["stable"]["gates"].values())


def test_p4cr_wrong_scanner_identity_fails_closed(tmp_path: Path) -> None:
    payload = CONTRACT.read_text(encoding="utf-8").replace(
        '"required_decision": "FAIL_CLOSED_BEFORE_CONFIRMATION_MATRIX_ENVELOPE"',
        '"required_decision": "wrong"',
    )
    path = tmp_path / "contract.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
