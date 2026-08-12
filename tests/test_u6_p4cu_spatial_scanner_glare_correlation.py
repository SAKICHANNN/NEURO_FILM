from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.spatial_scanner_glare_correlation import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cu_spatial_scanner_glare_correlation_v1.json"


def test_p4cu_contract_binds_closed_constant_flare() -> None:
    contract = load_contract(ROOT, CONTRACT)
    assert contract["experiment_id"] == "u6.p4cu-spatial-scanner-glare-correlation-v1"
    assert sum(contract["simulation"]["field_weights"]) == 1.0


def test_p4cu_spatial_field_control_is_discriminating() -> None:
    report = evaluate(ROOT, CONTRACT)
    errors = report["stable"]["confirmation_median_errors"]
    assert errors["correct_field"] < errors["uncorrected_field"]
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_p4cu_parent_drift_fails_closed(tmp_path: Path) -> None:
    payload = CONTRACT.read_text(encoding="utf-8").replace(
        '"required_decision": "close_current_additive_scanner_flare_control_without_tuning"',
        '"required_decision": "wrong"',
    )
    path = tmp_path / "contract.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
