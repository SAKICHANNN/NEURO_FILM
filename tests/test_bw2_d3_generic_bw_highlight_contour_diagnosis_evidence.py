from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/BW2_D3_GENERIC_BW_HIGHLIGHT_CONTOUR_DIAGNOSIS_RESULT.json"
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


def test_evidence_fails_closed_without_parameter_or_claim_rescue() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_NO_SINGLE_STAGE_CAUSE_IDENTIFIED"
    assert evidence["automatic_result"]["render_count"] == 30
    assert evidence["automatic_result"]["network_reads"] == 0
    assert evidence["decision_summary"] == {
        "necessary_stage_count": 0,
        "confirmed_fail_rows_cleared_by_any_single_variant": 0,
        "safe_product_parameter_change_opened": False,
        "separate_repair_leaf_opened": False,
    }
    assert all(not row["necessary_stage"] for row in evidence["variant_adjudications"])
    assert evidence["retained_boundaries"]["generic_bw_product_parameters_changed"] is False
    assert evidence["retained_boundaries"]["named_hp5_or_tri_x_claim_opened"] is False
    assert evidence["retained_boundaries"]["multi_stock_completion"] is False
    assert "not a repaired candidate" in evidence["claim_ceiling"]
