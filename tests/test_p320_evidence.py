from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P320_R1GS_HASSELBLAD_X2D_NO_COPY_INTAKE_RESULT.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p320_evidence_binds_fail_closed_reports_and_claim_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_R1GS_HASSELBLAD_X2D_NO_COPY_INTAKE"
    for key in ("config", "audit", "forward_report", "reverse_report"):
        binding = evidence["bindings"][key]
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha(path) == binding["sha256"]
    result = evidence["formal_result"]
    assert result["reports_byte_exact"] is True
    assert result["gates_passed"] == 8
    assert result["gates_total"] == 9
    assert result["failed_gate"] == "truncation-controls-reject"
    controls = {
        row["id"]: row["truncate_whole_file_one_byte_rejected"]
        for row in result["rows"]
    }
    assert controls == {
        "x2d_3fr_a": False,
        "x2d_fff_a": True,
        "x2d_3fr_b": False,
        "x2d_fff_b": True,
    }
    assert evidence["integrity"]["post_result_source_control_or_gate_change"] is False
    assert evidence["rights_and_product"]["producer_r1gs_pass_changed"] is False
    assert evidence["rights_and_product"]["product_mapping"] is False
    assert evidence["rights_and_product"]["candidate_3"] is False
