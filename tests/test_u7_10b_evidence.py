from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "evidence" / "U7_10B_INSTALLED_RUNTIME_DESKTOP_LAUNCH_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def test_u7_10b_evidence_binds_reports_visuals_and_stable_identity() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    execution = evidence["formal_execution"]
    reports = [
        ROOT / execution["forward_report_path"],
        ROOT / execution["reverse_report_path"],
    ]
    assert evidence["automatic_pass"] is True
    assert reports[0].read_bytes() == reports[1].read_bytes()
    assert all(
        path.stat().st_size == execution["report_bytes_each"] for path in reports
    )
    assert {_sha256(path) for path in reports} == {execution["report_sha256"]}
    report = json.loads(reports[0].read_text("utf-8"))
    assert report["status"] == evidence["status"]
    assert report["source_commit"] == execution["source_commit"]
    assert all(report["gates"].values())
    stable = report.pop("stable_identity")
    calculated = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert stable == calculated == execution["stable_identity"]

    visuals = [
        ROOT / execution["forward_visual_path"],
        ROOT / execution["reverse_visual_path"],
    ]
    assert visuals[0].read_bytes() == visuals[1].read_bytes()
    assert all(
        path.stat().st_size == execution["visual_bytes_each"] for path in visuals
    )
    assert {_sha256(path) for path in visuals} == {execution["visual_sha256"]}
    with Image.open(visuals[0]) as opened:
        assert opened.size == tuple(evidence["desktop_interaction"]["client_geometry"])


def test_u7_10b_evidence_binds_current_sources_and_parents() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    for binding in evidence["source_bindings"].values():
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]
    for binding in evidence["parent_evidence"].values():
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]
    assert _git("cat-file", "-t", evidence["formal_execution"]["source_commit"]) == (
        "commit"
    )
    assert all(evidence["gates"].values())


def test_u7_10b_excluded_attempts_and_claim_ceiling_remain_narrow() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    assert len(evidence["excluded_attempts"]) == 2
    assert all(row["accepted"] is False for row in evidence["excluded_attempts"])
    assert all(row["reports_written"] == 0 for row in evidence["excluded_attempts"])
    assert all(
        _git("cat-file", "-t", row["commit"]) == "commit"
        for row in evidence["excluded_attempts"]
    )
    claim = evidence["claim"]
    assert claim["mode"] == "film-inspired"
    assert claim["evidence_grade"] == "look-approximation"
    assert claim["calibrated_stock_response"] is False
    assert claim["physical_film_reproduction"] is False
    assert claim["public_release"] is False
    assert claim["standalone_installer"] is False
