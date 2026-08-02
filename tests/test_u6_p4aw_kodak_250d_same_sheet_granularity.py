from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.kodak_250d_same_sheet_granularity import (
    SameSheetGranularityError,
    evaluate_compatibility,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/u6_p4aw_kodak_250d_same_sheet_granularity_compatibility_v1.json"
)
PARENT_REPORT = (
    ROOT
    / "outputs/experiments/u6_p4av_kodak_250d_granularity_source_v1/report_a.json"
)


def test_contract_rejects_relaxed_control_gate(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["minimum_improvement_vs_wrong_channel_p5j"] = -1.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SameSheetGranularityError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not PARENT_REPORT.is_file(), reason="P4AV report unavailable")
def test_exact_same_sheet_experiment_is_repeatable() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_compatibility(contract, ROOT)
    second = evaluate_compatibility(contract, ROOT)
    assert first == second
    assert first["split_counts"] == {
        "blue": {"development": 13, "confirmation": 6},
        "green": {"development": 17, "confirmation": 9},
        "red": {"development": 18, "confirmation": 9},
    }
    assert first["gate_results"]["spatial_energy_convergence"]
    assert first["channel_energy_primary"]["blue"] > 1.0
    assert first["channel_energy_primary"]["red"] < 1.0


def test_parent_hash_tamper_fails_before_scoring(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["parents"]["p5j_decision_sha256"] = "0" * 64
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract = load_contract(path)
    with pytest.raises(SameSheetGranularityError, match="parent integrity"):
        evaluate_compatibility(contract, ROOT)
