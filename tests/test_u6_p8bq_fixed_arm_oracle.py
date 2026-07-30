from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_fixed_arm_oracle import evaluate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8bq_fixed_arm_oracle_diagnostic_v1.json"


def test_fixed_arm_oracle_binds_one_compiled_candidate() -> None:
    report = evaluate(ROOT, json.loads(CONFIG.read_text(encoding="utf-8")))
    assert report["candidate_identity"] == {
        "development_arm": "gauged_spatial_4000",
        "fresh_arm": "fixed_native_standard_full_strength_1_0",
        "bundle_sha256": (
            "3a14cb3c9257cf1a7dade92a03b15f1af2f74ba74c06233f0e9732fe536237af"
        ),
        "identity_chain_valid": True,
    }


def test_development_oracle_is_descriptive_but_not_fresh_evidence() -> None:
    report = evaluate(ROOT, json.loads(CONFIG.read_text(encoding="utf-8")))
    development = report["development"]
    assert development["global_observed_score"] == 13
    assert development["physical_observed_score"] == 14
    assert development["oracle_observed_score"] == 20
    assert development["oracle_gain_votes_over_global"] == 7
    assert development["physical_selected_source_count"] == 7
    assert development["censored_vote_count"] == 0


def test_fresh_oracle_gain_is_censored_and_cannot_open_router() -> None:
    report = evaluate(ROOT, json.loads(CONFIG.read_text(encoding="utf-8")))
    fresh = report["fresh_confirmation"]
    assert fresh["global_observed_score"] == 7
    assert fresh["physical_observed_score"] == 5
    assert fresh["oracle_observed_score"] == 9
    assert fresh["oracle_gain_votes_over_global"] == 2
    assert fresh["physical_selected_source_count"] == 2
    assert fresh["censored_vote_count"] == 15
    assert fresh["censored_vote_fraction"] == pytest.approx(15 / 27)
    assert report["gate_checks"] == {
        "fresh_observed_gain_votes": True,
        "fresh_observed_gain_fraction": True,
        "fresh_selected_source_support": True,
        "complete_pairwise_observation": False,
    }
    assert report["decision"] == "censored_no_router"
    assert report["router_evidence_pass"] is False
    assert report["router_opened"] is False
    assert report["production_default_changed"] is False


def test_report_is_repeat_exact_and_parent_hash_drift_fails() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = evaluate(ROOT, config)
    second = evaluate(ROOT, config)
    assert first == second
    assert len(first["stable_evidence_id"]) == 64
    config["parents"]["p8bp_fresh_report"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent hash mismatch"):
        evaluate(ROOT, config)
