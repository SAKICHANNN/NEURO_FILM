from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/BW2_D2_GENERIC_BW_POPULATION_SEVERE_REVIEW_RESULT.json"
)


def test_evidence_binds_exact_contract_code_runner_test_and_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for binding in evidence["bindings"].values():
        assert hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest() == binding[
            "sha256"
        ]
    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert len(forward.read_bytes()) == reports["bytes_each"]
    assert hashlib.sha256(forward.read_bytes()).hexdigest() == reports["sha256"]


def test_evidence_fails_visual_gate_without_inflating_claim() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_CONFIRMED_HIGHLIGHT_CONTOUR_ARTIFACTS"
    assert evidence["automatic_result"]["failed_gates"] == []
    assert evidence["decision_counts"] == {
        "PASS_NO_CONFIRMED_SEVERE_ARTIFACT": 13,
        "FAIL_CONFIRMED_SEVERE_ARTIFACT": 3,
        "REVIEW_UNRESOLVED": 0,
    }
    failed = [
        row["source_id"]
        for row in evidence["source_adjudications"]
        if row["decision"] == "FAIL_CONFIRMED_SEVERE_ARTIFACT"
    ]
    assert failed == ["sony_dslr_a290", "olympus_e_p7", "leica_d_lux_6"]
    assert evidence["retained_boundaries"]["multi_stock_completion"] is False
    assert evidence["retained_boundaries"]["named_hp5_or_tri_x_claim_opened"] is False
    assert "not HP5 or Tri-X stock response" in evidence["claim_ceiling"]
