from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.analytic_y_chromaticity_adjudication import (
    AnalyticYChromaticityAdjudicationError,
)
from src.eval.analytic_y_chromaticity_third_adjudication import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb74_analytic_y_chromaticity_adjudication_v1.json"
EVIDENCE = (
    ROOT
    / "docs/evidence/U5_R2CB74_ANALYTIC_Y_CHROMATICITY_THIRD_CONFIRMATION_RESULT.json"
)


def test_cb74_adjudication_contract_binds_frozen_observations() -> None:
    contract = load_contract(CONTRACT)
    observations = ROOT / contract["evidence"]["observations_path"]
    payload = json.loads(observations.read_text(encoding="utf-8"))
    assert payload["mapping_opened"] is False
    assert len(payload["source_order"]) == 9
    assert contract["gates"]["minimum_candidate_aggregate_choices"] == 18


def test_cb74_adjudication_rejects_contract_identity_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["experiment_id"] = "U5.R2CB74-drift"
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AnalyticYChromaticityAdjudicationError, match="contract drift"):
        load_contract(path)


def test_cb74_decision_closes_without_rescue() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["repeat_report_and_output_hashes_exact"] is True
    assert evidence["automatic_pass"] is True
    assert evidence["autonomous_visual_metrics"]["candidate_round_choices"] == [4, 4, 7]
    assert evidence["decision"] == "close_cb74_without_rescue_retain_ao6"
    assert evidence["product_default_changed"] is False
