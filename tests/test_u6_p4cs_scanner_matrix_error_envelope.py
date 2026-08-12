from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.scanner_matrix_error_envelope import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cs_scanner_matrix_error_envelope_v1.json"


def test_p4cs_contract_binds_p4cr_and_fixed_grid() -> None:
    contract = load_contract(ROOT, CONTRACT)
    assert contract["experiment_id"] == "u6.p4cs-scanner-matrix-error-envelope-v1"
    assert contract["simulation"]["relative_frobenius_error_levels"] == [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04]
    assert len(contract["simulation"]["zero_row_sum_perturbation_directions"]) == 3


def test_p4cs_selects_development_envelope_before_confirmation() -> None:
    report = evaluate(ROOT, CONTRACT)
    stable = report["stable"]
    assert stable["development_selected_level_index"] >= 0
    assert stable["development_selected_relative_frobenius_error"] in [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04]
    assert report["automatic_pass"] is all(stable["gates"].values())


def test_p4cs_parent_drift_fails_closed(tmp_path: Path) -> None:
    payload = CONTRACT.read_text(encoding="utf-8").replace(
        '"required_decision": "retain_scanner_unmixing_as_required_layer_correlation_identifiability_control"',
        '"required_decision": "wrong"',
    )
    path = tmp_path / "contract.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
