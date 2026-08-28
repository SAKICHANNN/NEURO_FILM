from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2G_GENERIC_BW_SEVERE_VETO_RESULT.json"


def test_u7_2g_evidence_binds_implementation_and_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    for binding in evidence["bindings"].values():
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    assert len(forward.read_bytes()) == reports["bytes_each"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert hashlib.sha256(forward.read_bytes()).hexdigest() == reports["sha256"]
    assert evidence["execution_result"]["all_ten_formal_gates_pass"] is True
    assert evidence["catalog_result"]["generic_bw_availability"] == (
        "blocked_severe_artifact"
    )
