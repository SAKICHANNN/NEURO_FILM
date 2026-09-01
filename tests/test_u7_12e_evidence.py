from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_12E_DESKTOP_BATCH_RECOVERY_UI_RESULT.json"


def _git_blob_sha256(path: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"4793eff6:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def test_u7_12e_evidence_is_fail_closed_without_mechanical_drift() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_U7_12E_DESKTOP_BATCH_RECOVERY_UI"
    assert evidence["formal_reports"]["full_reports_byte_exact"]
    assert evidence["gates"]["focused_parent_tests_pass"] is False
    assert sum(not value for value in evidence["gates"].values()) == 1
    assert evidence["mechanical_result"]["fresh_session_resume_completed"]
    assert evidence["mechanical_result"]["recovered_and_u7_11a_members_exact"]
    assert evidence["mechanical_result"]["recovered_and_u7_11a_receipt_exact"]
    assert evidence["focused_parent_test_diagnostic"]["isolated_exact_test_result"] == "1 passed"
    assert evidence["claim"]["formal_ui_integration_pass"] is False
    assert evidence["claim"]["u7_12d_core_or_u7_11a_outputs_invalidated"] is False


def test_u7_12e_evidence_binds_formal_git_objects() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    for path, expected in evidence["bindings"]["bound_git_blob_sha256"].items():
        assert _git_blob_sha256(path) == expected
