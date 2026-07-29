from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.ao6_adaptive_style_dose import (
    evaluate_adaptive_style_dose,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2at0_ao6_adaptive_style_dose_v1.json"
DECISION = (
    ROOT / "configs/u5_r2at0_ao6_adaptive_style_dose_decision_v1.json"
)
CONFIG_SHA256 = "aec853877133288b6f0618c1ee97d136c2202a9b6e7105747ccf0b1d3db5d5f9"


def test_frozen_adaptive_style_dose_passes() -> None:
    report = evaluate_adaptive_style_dose(
        root=ROOT,
        contract=load_contract(CONFIG),
        contract_sha256=CONFIG_SHA256,
    )
    assert report["automatic_passed"]
    assert report["automatic_decision"] == "open_development_visual_gate"
    assert report["metrics"]["selected_strength_level_count"] == 4
    assert report["metrics"]["dose_mad_ratio_vs_fixed_global"] < 0.5
    assert all(report["checks"].values())


def test_selection_is_deterministic_and_hash_bound() -> None:
    contract = load_contract(CONFIG)
    first = evaluate_adaptive_style_dose(
        root=ROOT,
        contract=contract,
        contract_sha256=CONFIG_SHA256,
    )
    second = evaluate_adaptive_style_dose(
        root=ROOT,
        contract=contract,
        contract_sha256=CONFIG_SHA256,
    )
    assert first == second
    assert len(first["selections"]) == 41
    assert all(len(row["output_sha256"]) == 64 for row in first["selections"])


def test_content_or_refit_drift_fails_closed() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract["policy"]["content_features_allowed"] = True
    with pytest.raises(ValueError, match="boundary drift"):
        evaluate_adaptive_style_dose(
            root=ROOT,
            contract=contract,
            contract_sha256=CONFIG_SHA256,
        )


def test_parent_hash_drift_fails_closed() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract["parent_automatic_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_adaptive_style_dose(
            root=ROOT,
            contract=contract,
            contract_sha256=CONFIG_SHA256,
        )


def test_frozen_decision_closes_visual_policy_without_safety_failure() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    visual = decision["autonomous_visual_result"]
    assert decision["decision"] == "close_adaptive_style_dose_target_2p5"
    assert decision["automatic_result"]["passed"]
    assert not visual["passed"]
    assert visual["adaptive_wins_vs_fixed_ao6"] == 0
    assert visual["confirmed_severe_artifacts"] == 0
    assert all(
        row["ranking_before_reveal"]
        for row in visual["rounds"]
    )
