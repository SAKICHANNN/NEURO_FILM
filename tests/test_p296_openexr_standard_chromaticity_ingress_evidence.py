from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P296_OPENEXR_STANDARD_CHROMATICITY_INGRESS_RESULT.json"


def test_p296_evidence_closes_all_three_frozen_cross_source_gates() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_OPENEXR_STANDARD_CHROMATICITY_INGRESS"
    assert (
        evidence["results"]["cross_source_rmse"]
        > evidence["frozen_thresholds"]["maximum_rmse"]
    )
    assert (
        evidence["results"]["cross_source_p99_absolute_error"]
        > evidence["frozen_thresholds"]["maximum_p99_absolute_error"]
    )
    assert (
        evidence["results"]["cross_source_maximum_absolute_error"]
        > evidence["frozen_thresholds"]["maximum_absolute_error"]
    )
    assert not evidence["gates"]["cross_source_rmse"]
    assert not evidence["gates"]["cross_source_p99"]
    assert not evidence["gates"]["cross_source_maximum"]


def test_p296_evidence_preserves_mechanical_and_claim_boundaries() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["formal"]["reports_byte_exact"]
    assert evidence["gates"]["invalid_controls_atomic"]
    assert evidence["gates"]["zero_new_boundary"]
    assert evidence["candidate_count_after_result"] == "2/3"
    assert evidence["consumer_mapping"] is False
    assert "unexported and unintegrated" in evidence["interpretation"]
