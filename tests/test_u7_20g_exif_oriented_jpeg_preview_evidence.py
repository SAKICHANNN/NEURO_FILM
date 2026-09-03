from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_20G_EXIF_ORIENTED_JPEG_PREVIEW_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_20g_evidence_binds_reports_and_claim() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_U7_20G_EXIF_ORIENTED_JPEG_PREVIEW"
    assert evidence["automatic_pass"] is True
    reports = evidence["formal_execution"]["reports"]
    assert len(reports) == 2
    assert reports[0]["sha256"] == reports[1]["sha256"]
    for row in reports:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert _sha256(path) == row["sha256"]
    assert all(evidence["gates"].values())
    assert evidence["claim_ceiling"]["evidence_grade"] == "look-approximation"
    assert evidence["claim_ceiling"]["calibrated_stock_response"] is False


def test_u7_20g_evidence_binds_committed_sources() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for relative, binding in evidence["source_bindings"].items():
        assert binding["bytes"] > 0
        assert_historical_evidence_binding(
            ROOT,
            {"path": relative, **binding},
        )
