from __future__ import annotations

import json
from pathlib import Path

from src.eval.global_frontier import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs/u6_p8j_context_first_resources_decision_v1.json"
REPORT = ROOT / "outputs/u6_p8j_context_first_resources_v1/run_a/report.json"


def test_p8j_decision_binds_exact_evidence() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert decision["report_sha256"] == sha256_file(REPORT)
    assert (
        decision["stable_evidence_id"]
        == report["stable_evidence_id"]
    )
    assert decision["all_runs_successful"]
    assert decision["all_repeat_identity_exact"]
    assert decision["same_output_identity_as_p8d_through_p8i"]


def test_p8j_is_only_a_prerequisite_not_a_resource_win() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))

    assert not decision["standalone_resource_improvement"]
    assert decision["retained_only_for_next_combination"]
    assert not decision["performance_target_pass"]
    assert not decision["production_default_changed"]
    assert not decision["calibration_claim_opened"]
    assert not decision["native_runtime_opened"]
    assert decision["next_leaf"].startswith("U6.P8K")
