from pathlib import Path

import pytest

from src.eval.structured_scene_scanner_flare import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cw_structured_scene_scanner_flare_v1.json"


def test_p4cw_contract_binds_p4cv() -> None:
    assert (
        load_contract(ROOT, CONTRACT)["experiment_id"]
        == "u6.p4cw-structured-scene-scanner-flare-v1"
    )


def test_p4cw_result_is_finite_and_gate_consistent() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert all(
        value >= 0 for value in report["stable"]["confirmation_median_errors"].values()
    )
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_p4cw_parent_drift_fails_closed(tmp_path: Path) -> None:
    payload = CONTRACT.read_text(encoding="utf-8").replace(
        '"required_decision": "close_current_signal_dependent_scanner_flare_control_without_tuning"',
        '"required_decision": "wrong"',
    )
    path = tmp_path / "contract.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
