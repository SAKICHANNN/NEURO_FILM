from __future__ import annotations

import copy
import json
from pathlib import Path

from src.eval.analytic_y_chromaticity_adjudication import adjudicate

ROOT = Path(__file__).resolve().parents[1]
DECISION = (
    ROOT / "configs/u5_r2cb50_analytic_y_chromaticity_development_decision_v1.json"
)


def test_cb50_adjudicator_counts_sealed_choices(monkeypatch, tmp_path) -> None:
    source_ids = ["s1", "s2", "s3"]
    report = {
        "automatic_pass": True,
        "stable_evidence_id": "stable",
        "rows": [{"id": value} for value in source_ids],
        "blind_sheets": [{"round": 1, "sha256": "sheet"}],
        "sealed_mappings": {"1": {value: ["candidate", "ao6"] for value in source_ids}},
    }
    observations = {
        "report_sha256": "report",
        "mapping_opened": False,
        "source_order": source_ids,
        "rounds": [{"round": 1, "sheet_sha256": "sheet", "choices": ["A", "B", "A"]}],
        "severe_review": {"confirmed_severe_artifact_count": 0},
    }
    values = iter([report, observations])
    monkeypatch.setattr(
        "src.eval.analytic_y_chromaticity_adjudication._load_exact",
        lambda *_: copy.deepcopy(next(values)),
    )
    config = {
        "evidence": {
            "report_sha256": "report",
            "stable_evidence_id": "stable",
            "observations_path": "observations.json",
            "observations_sha256": "observations",
        },
        "gates": {
            "maximum_confirmed_severe_artifact_count": 0,
            "minimum_candidate_round_wins": 1,
            "minimum_candidate_aggregate_choices": 2,
            "minimum_candidate_source_majorities": 2,
        },
        "claim_ceiling": "test",
    }
    result = adjudicate(config, tmp_path, tmp_path / "report.json")
    assert result["round_candidate_choices"] == [2]
    assert result["candidate_aggregate_choices"] == 2
    assert result["candidate_source_majorities"] == 0
    assert result["pass"] is False


def test_cb50_decision_opens_only_source_disjoint_confirmation() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["repeat_report_and_output_hashes_exact"] is True
    assert decision["automatic_metrics"]["maximum_lstar_inversion_fraction"] == 0
    assert decision["autonomous_visual_metrics"]["candidate_round_choices"] == [8, 8, 8]
    assert (
        decision["decision"]
        == "pass_cb50_development_open_source_disjoint_confirmation"
    )
    assert "not independent confirmation" in decision["claim_ceiling"]
