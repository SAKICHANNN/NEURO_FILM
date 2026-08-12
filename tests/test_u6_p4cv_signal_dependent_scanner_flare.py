from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.signal_dependent_scanner_flare import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cv_signal_dependent_scanner_flare_v1.json"


def test_p4cv_contract_binds_p4cu() -> None:
    contract = load_contract(ROOT, CONTRACT)
    assert contract["experiment_id"] == "u6.p4cv-signal-dependent-scanner-flare-v1"
    assert contract["simulation"]["gaussian_sigma_pixels"] == 18.0


def test_p4cv_signal_dependent_control_is_finite_and_gate_consistent() -> None:
    report = evaluate(ROOT, CONTRACT)
    errors = report["stable"]["confirmation_median_errors"]
    assert all(value >= 0.0 for value in errors.values())
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_p4cv_parent_drift_fails_closed(tmp_path: Path) -> None:
    payload = CONTRACT.read_text(encoding="utf-8").replace(
        '"required_decision": "close_current_spatial_scanner_glare_control_without_tuning"',
        '"required_decision": "wrong"',
    )
    path = tmp_path / "contract.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
