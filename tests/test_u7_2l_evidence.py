from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2L_PRODUCT_CREATE_ONLY_RENDER_TRANSACTION_RESULT.json"


def test_u7_2l_evidence_binds_create_only_product_result() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    for binding in evidence["bindings"].values():
        assert_historical_evidence_binding(ROOT, binding)

    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert len(forward.read_bytes()) == reports["bytes_each"]
    assert hashlib.sha256(forward.read_bytes()).hexdigest() == reports["sha256"]

    result = evidence["result"]
    assert result["formal_gate_count"] == result["formal_gate_pass_count"] == 15
    assert result["concurrent_success_count"] == 1
    assert result["concurrent_fail_closed_count"] == 1
    assert result["owned_stage_residue_count"] == 0
    assert evidence["retained_boundaries"]["calibrated_stock_or_stock_response_claim"] is False
