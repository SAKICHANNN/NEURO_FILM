from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.gradient_budgeted_fraction_confirmation import (
    GradientBudgetedFractionConfirmationError,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb34_gradient_budgeted_fraction_confirmation_v1.json"
DECISION = (
    ROOT / "configs/u5_r2cb34_gradient_budgeted_fraction_confirmation_decision_v1.json"
)


def test_cb34_contract_freezes_exact_cb33_and_disjoint_population() -> None:
    contract = load_contract(CONTRACT)
    assert contract["experiment_id"] == "U5.R2CB34"
    assert contract["parents"]["cb33_required_decision"] == (
        "pass_development_gold_stress_open_source_disjoint_confirmation"
    )
    assert contract["population"]["source_count_exact"] == 12
    assert contract["population"]["exact_decoded_sha_overlap_with_cb27_cb28_cb29_u41"] == 0
    assert contract["operator"]["dose_grid"][-1] == 0.0
    assert contract["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"] == 1.35


def test_cb34_rejects_parent_decision_drift(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["parents"]["cb33_required_decision"] = "forged"
    with pytest.raises(GradientBudgetedFractionConfirmationError, match="decision drift"):
        evaluate(contract, ROOT, tmp_path / "out")


def test_cb34_decision_closes_before_visual_review() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["report_sha256"] == (
        "2a80ad0b5834e3dbffa7392a3f0a0276ca3a1c1bcf3d29f006433ca9394434a3"
    )
    assert decision["failed_checks"] == ["gradient_order"]
    assert decision["visual_review_status"] == "forbidden_by_automatic_gate"
    assert decision["decision"] == (
        "close_exact_cb33_on_source_disjoint_lstar_order_failure"
    )
