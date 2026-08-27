from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P263_R1EJ_DNG_PROFILE_GAIN_TABLE_CALLABLE_INTAKE_RESULT.json"


def test_p263_evidence_is_fail_closed_and_exact() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_R1EJ_DNG_PROFILE_GAIN_TABLE_CALLABLE_INTAKE"
    assert all(evidence["gates"].values())
    formal = evidence["formal_execution"]
    reports = [ROOT / formal["forward_report"], ROOT / formal["reverse_report"]]
    assert reports[0].read_bytes() == reports[1].read_bytes()
    assert len(reports[0].read_bytes()) == formal["report_bytes"]
    assert hashlib.sha256(reports[0].read_bytes()).hexdigest() == formal["report_sha256"]
    assert formal["network_reads"] == 0
    assert formal["new_raw_dng_pixel_target_reads"] == 0
    assert formal["temporary_residue"] == 0
