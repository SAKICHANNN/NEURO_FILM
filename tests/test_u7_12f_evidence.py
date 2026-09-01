from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_12F_DESKTOP_ERROR_SELECTION_RESET_RESULT.json"


def _git_blob_sha256(commit: str, path: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def test_u7_12f_evidence_records_pass_and_excluded_identity_diagnostic() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_U7_12F_DESKTOP_ERROR_SELECTION_RESET"
    assert evidence["formal_reports"]["full_reports_byte_exact"]
    assert all(evidence["gates"].values())
    assert all(evidence["fresh_process_cases"].values())
    assert evidence["trigger_binding"]["parent_clears_style"] is False
    assert evidence["trigger_binding"]["current_clear_count"] == 1
    excluded = evidence["excluded_first_formal_attempt"]
    assert excluded["status"] == "FAIL_CLOSED_U7_12F_DESKTOP_ERROR_SELECTION_RESET"
    assert excluded["product_code_tests_or_gates_changed"] is False
    equivalence = excluded["verified_equivalence"]
    assert (
        equivalence["parent_windows_checkout_sha256"]
        == equivalence["frozen_expected_windows_checkout_sha256"]
    )


def test_u7_12f_evidence_binds_formal_git_objects() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    commit = evidence["source_commit"]
    for path, expected in evidence["bindings"]["bound_git_blob_sha256"].items():
        assert _git_blob_sha256(commit, path) == expected
