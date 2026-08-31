from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.inference.render_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_8F_WINDOWS_SAFE_BATCH_JOB_ID_RESULT.json"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def test_evidence_binds_formal_reports_and_git_objects() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == "PASS_PRIVATE_U7_8F_WINDOWS_SAFE_BATCH_JOB_ID"
    commit = evidence["bindings"]["formal_execution_commit"]
    assert _git("cat-file", "-t", commit) == "commit"
    for relative, binding in evidence["bindings"]["files"].items():
        assert _git("rev-parse", f"{commit}:{relative}") == binding["git_blob"]
        assert int(_git("cat-file", "-s", binding["git_blob"])) == binding["bytes"]
        payload = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT
        )
        import hashlib

        assert hashlib.sha256(payload).hexdigest() == binding["sha256"]
    forward = ROOT / evidence["formal_reports"]["forward"]["path"]
    reverse = ROOT / evidence["formal_reports"]["reverse"]["path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert sha256_file(forward) == evidence["formal_reports"]["forward"]["sha256"]
    assert sha256_file(reverse) == evidence["formal_reports"]["reverse"]["sha256"]
    report = json.loads(forward.read_text(encoding="utf-8"))
    assert report["bindings"]["execution_commit"] == commit
    assert (
        report["scientific_identity"]
        == evidence["formal_reports"]["scientific_identity"]
    )


def test_evidence_records_complete_fail_closed_scope() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert all(evidence["gate_results"].values())
    assert evidence["metrics"] == {
        "valid_control_count": 4,
        "invalid_control_count": 10,
        "entrypoint_control_count": 20,
        "forbidden_operation_call_count": 0,
        "network_requests": 0,
        "owned_scratch_residue_count": 0,
    }
    assert evidence["historical_evidence"]["u7_8c"].endswith("not rerun")
    assert evidence["historical_evidence"]["u7_8e"].endswith("not rerun")
