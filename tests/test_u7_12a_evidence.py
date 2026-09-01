from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/U7_12A_DESKTOP_FOREGROUND_WORKER_CLOSE_SAFETY_RESULT.json"
)
EVIDENCE_SHA256 = "efb97ebfe819d8590f40f8e8bc2b1f82a7b6c647580f5ff8cd1b16ce43655d7b"


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_u7_12a_evidence_is_exact_pass() -> None:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EVIDENCE_SHA256
    report = json.loads(payload)
    assert report["status"] == "PASS_PRIVATE_U7_12A_DESKTOP_FOREGROUND_CLOSE_SAFETY"
    assert all(report["gates"].values())
    assert all(row["passed"] for row in report["case_results"].values())
    assert report["claim"] == {
        "batch_changed": False,
        "calibrated_stock_response": False,
        "evidence_grade": "look-approximation",
        "installer_changed": False,
        "mode": "film-inspired",
        "physical_film_reproduction": False,
        "private_windows_tk_close_safety_only": True,
        "public_release": False,
        "renderer_changed": False,
    }


def test_u7_12a_evidence_binds_formal_git_objects() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    commit = report["source_commit"]
    for path, expected in report["bindings"].items():
        assert hashlib.sha256(_git_blob(commit, path)).hexdigest() == expected

