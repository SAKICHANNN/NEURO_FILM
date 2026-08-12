from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.scanner_flare_layer_correlation import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ct_scanner_flare_layer_correlation_v1.json"


def test_p4ct_contract_binds_p4cs() -> None:
    contract = load_contract(ROOT, CONTRACT)
    assert contract["experiment_id"] == "u6.p4ct-scanner-flare-layer-correlation-v1"
    assert contract["simulation"]["wrong_flare_scale"] == 0.5


def test_p4ct_flare_control_is_discriminating() -> None:
    report = evaluate(ROOT, CONTRACT)
    errors = report["stable"]["confirmation_median_errors"]
    assert errors["correct_flare"] < errors["wrong_flare"] < errors["uncorrected_flare"]
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_p4ct_parent_drift_fails_closed(tmp_path: Path) -> None:
    payload = CONTRACT.read_text(encoding="utf-8").replace(
        '"required_decision": "retain_frozen_scanner_matrix_error_envelope_as_layer_correlation_admission_control"',
        '"required_decision": "wrong"',
    )
    path = tmp_path / "contract.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
