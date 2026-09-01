from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_12D_DESKTOP_SINGLE_LOOK_BATCH_RECOVERY_RESULT.json"
EVIDENCE_SHA256 = "f052be6957f95f2029728c12d7946d29263d031b0216c3f9c80b7188400db58c"


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_u7_12d_evidence_is_exact_private_pass() -> None:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EVIDENCE_SHA256
    report = json.loads(payload)
    assert report["status"] == (
        "PASS_PRIVATE_U7_12D_DESKTOP_SINGLE_LOOK_BATCH_RECOVERY"
    )
    assert report["scientific_identity"] == (
        "sha256:2605762415427f3343b08aa3bdde090567d0395c321224d83bd0ee5872110ac7"
    )
    assert all(report["gates"].values())
    assert report["formal_reports"]["full_reports_byte_exact"]
    assert report["formal_reports"]["forward"] == report["formal_reports"]["reverse"]
    assert report["renderer_calls"] == {
        "pause": 3,
        "resume": 5,
        "uninterrupted_control": 8,
    }
    assert report["excluded_first_formal_attempt"]["science_or_gates_changed"] is False


def test_u7_12d_evidence_binds_formal_git_objects() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    commit = report["bindings"]["formal_source_commit"]
    assert commit == report["source_commit"]
    for path, expected in report["bindings"]["files"].items():
        assert hashlib.sha256(_git_blob(commit, path)).hexdigest() == expected


def test_u7_12d_claim_ceiling_remains_look_approximation() -> None:
    claim = json.loads(EVIDENCE.read_text("utf-8"))["claim"]
    assert claim["mode"] == "film-inspired"
    assert claim["evidence_grade"] == "look-approximation"
    assert claim["private_windows_python_restart_recovery_only"]
    assert claim["calibrated_stock_response"] is False
    assert claim["physical_film_reproduction"] is False
    assert claim["stock_distinguishability"] is False
