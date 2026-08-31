from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U4_5E_RECEIPT_BOUND_PREVIEW_SESSION_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evidence_binds_both_reports_and_exact_scientific_payload() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    reports = evidence["formal_reports"]
    loaded = []
    for key in ("forward", "reverse"):
        binding = reports[key]
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["status"] == evidence["status"]
        assert report["scientific_stable_id"] == reports["scientific_stable_id"]
        assert all(report["scientific"]["gates"].values())
        assert all(
            count == 0
            for count in report["scientific"]["warm_operation_counts"].values()
        )
        loaded.append(report)
    assert loaded[0]["scientific"] == loaded[1]["scientific"]
    assert reports["whole_report_byte_exact"] is False
    assert reports["difference_scope"] == ["order", "timing"]


def test_evidence_preserves_claim_and_historical_boundaries() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["result"]["formal_gate_pass_count"] == 15
    assert evidence["result"]["warm_operation_attempts_total"] == 0
    assert evidence["result"]["invalid_receipt_controls_passed"] == 5
    assert evidence["result"]["cache_index_tamper_controls_passed"] == 6
    assert evidence["result"]["media_drift_controls_passed"] == 3
    assert all(value is False for value in evidence["retained_boundaries"].values())
    assert "Look Approximation" in evidence["claim_ceiling"]
    assert "physical stock response" in evidence["claim_ceiling"]
