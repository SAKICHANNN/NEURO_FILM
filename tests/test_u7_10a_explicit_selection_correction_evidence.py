from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/evidence/U7_10A_PRODUCT_DESKTOP_EXPLICIT_SELECTION_CORRECTION_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_10a_correction_binds_current_sources_and_formal_artifacts() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_U7_10A_PRODUCT_DESKTOP_INPUT_WORKFLOW"
    assert all(evidence["correction"].values())
    for binding in evidence["bindings"].values():
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]

    execution = evidence["execution"]
    reports = [
        ROOT / execution["forward_report_path"],
        ROOT / execution["reverse_report_path"],
    ]
    assert all(path.stat().st_size == execution["report_bytes"] for path in reports)
    assert {_sha256(path) for path in reports} == {execution["report_sha256"]}
    assert reports[0].read_bytes() == reports[1].read_bytes()
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["source_commit"] == execution["source_commit"]
    assert (
        report["scientific_identity_sha256"] == execution["scientific_identity_sha256"]
    )
    assert all(report["scientific"]["gates"].values())

    visual = evidence["visual"]
    screenshot = ROOT / visual["path"]
    assert screenshot.stat().st_size == visual["bytes"]
    assert _sha256(screenshot) == visual["sha256"]
    with Image.open(screenshot) as opened:
        assert opened.size == (visual["width"], visual["height"])
    assert report["scientific"]["gui_smoke"]["state"] == visual["automated_state"]


def test_u7_10a_correction_preserves_old_evidence_and_claim_ceiling() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    superseded = evidence["supersedes"]
    assert _sha256(ROOT / superseded["evidence_path"]) == superseded["evidence_sha256"]
    assert all(evidence["gates"].values())
    manual = evidence["visual"]["manual_review"]
    assert manual["explicit_selected_radio_visible"] is True
    assert manual["look_selected_status_visible"] is True
    assert manual["absolute_path_visible"] is False
    ceiling = evidence["claim_ceiling"].lower()
    assert "not calibrated stock response" in ceiling
    assert "not" in ceiling and "public installer" in ceiling
