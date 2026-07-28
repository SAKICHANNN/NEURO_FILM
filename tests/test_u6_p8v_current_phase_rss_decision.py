from __future__ import annotations

import json
from pathlib import Path

from src.eval.global_frontier import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs/u6_p8v_current_phase_rss_decision_v1.json"
REPORT = ROOT / "outputs/u6_p8v_current_phase_rss_v1/run_a/report.json"


def test_p8v_decision_binds_exact_phase_evidence() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert decision["report_sha256"] == sha256_file(REPORT)
    assert (
        decision["stable_evidence_id"]
        == report["stable_evidence_id"]
    )
    assert decision["all_runs_successful"]
    assert decision["all_output_identity_exact"]
    assert decision["dominant_phase_stable"]
    assert decision["dominant_phase"] == "physical-spatial"


def test_p8v_opens_one_bounded_tile_grid() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))

    assert decision["desktop_python_reference_memory_target_pass"]
    assert not decision["mobile_native_memory_target_pass"]
    assert not decision["performance_target_pass"]
    assert not decision["production_default_changed"]
    assert not decision["calibration_claim_opened"]
    assert not decision["native_runtime_opened"]
    assert decision["next_leaf"].startswith("U6.P8W")
