from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_peaked_density_shape import (
    PeakedDensityShapeError,
    evaluate_shape,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ac_peaked_density_grain_shape_v1.json"
REPORT = ROOT / "outputs/eval/u6_p4aa_nasa_density_grain_source_v1/report_run1.json"


def test_contract_freezes_source_stated_peak_and_no_fit() -> None:
    contract = load_contract(CONTRACT)
    assert contract["model"]["peak_density"] == 2.0
    assert not contract["model"]["amplitude_fit_allowed"]
    assert not contract["model"]["table_value_fit_allowed"]
    assert contract["gates"]["required_eligible_series"] == 9
    assert not contract["evaluation"]["post_result_retuning_allowed"]


def test_contract_rejects_peak_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["model"]["peak_density"] = 2.1
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PeakedDensityShapeError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not REPORT.is_file(), reason="P4AA report unavailable")
def test_peaked_shape_is_repeat_exact_and_reports_baseline() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_shape(contract, ROOT)
    second = evaluate_shape(contract, ROOT)
    assert first == second
    assert first["metrics"]["eligible_series"] == 9
    assert len(first["series"]) == 9
    assert first["metrics"]["p4d_median_normalized_relative_error"] > 0.0
    assert first["metrics"]["candidate_median_normalized_relative_error"] > 0.0
    assert first["stable_evidence_id"]


@pytest.mark.skipif(not REPORT.is_file(), reason="P4AA report unavailable")
def test_report_decision_is_exactly_derived_from_frozen_gates() -> None:
    report = evaluate_shape(load_contract(CONTRACT), ROOT)
    assert report["passed"] is all(report["gate_results"].values())
    assert report["decision"] == (
        "open_mean_preserving_peaked_compound_poisson_synthetic_leaf"
        if report["passed"]
        else "close_peaked_density_shape_without_rescue"
    )
