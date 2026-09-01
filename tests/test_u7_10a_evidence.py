from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_10A_PRODUCT_DESKTOP_INPUT_WORKFLOW_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_10a_evidence_binds_current_sources_and_accepted_artifacts() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_U7_10A_PRODUCT_DESKTOP_INPUT_WORKFLOW"

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
    assert (
        report["scientific"]["gui_smoke"]["ready_state_png_sha256"] == visual["sha256"]
    )


def test_u7_10a_evidence_preserves_claim_ceiling_and_excludes_invalid_attempt() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    manual_review = evidence["visual"]["manual_review"]
    assert manual_review["look_approximation_claims_visible"] is True
    assert manual_review["not_calibrated_disclaimer_visible"] is True
    assert manual_review["absolute_path_visible"] is False
    assert manual_review["overlap_or_crop_observed"] is False
    assert all(evidence["gates"].values())
    assert len(evidence["excluded"]) == 1
    excluded = evidence["excluded"][0]
    assert excluded["report_sha256"] != evidence["execution"]["report_sha256"]
    assert excluded["screenshot_sha256"] != evidence["visual"]["sha256"]
    ceiling = evidence["claim_ceiling"].lower()
    assert "not calibrated stock response" in ceiling
    assert "not" in ceiling and "public installer" in ceiling
