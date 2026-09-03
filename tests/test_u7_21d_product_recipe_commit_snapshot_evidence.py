from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_21D_PRODUCT_RECIPE_COMMIT_SNAPSHOT_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evidence_binds_reports_runtime_and_claim() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["automatic_pass"] is True
    assert evidence["status"] == "PASS_PRIVATE_U7_21D_PRODUCT_RECIPE_COMMIT_SNAPSHOT"
    reports = evidence["formal_execution"]["reports"]
    assert reports[0]["sha256"] == reports[1]["sha256"]
    for row in reports:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert _sha256(path) == row["sha256"]
    runtime = evidence["installed_runtime"]
    receipt = ROOT / runtime["path"] / "product-runtime.json"
    assert receipt.stat().st_size == runtime["receipt_bytes"]
    assert _sha256(receipt) == runtime["receipt_sha256"]
    assert json.loads(receipt.read_text(encoding="utf-8"))["source_commit"] == runtime["installed_source_commit"]
    assert all(evidence["gates"].values())
    assert evidence["claim_ceiling"]["calibrated_stock_response"] is False


def test_evidence_binds_committed_sources() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for relative, binding in evidence["source_bindings"].items():
        assert_historical_evidence_binding(ROOT, {"path": relative, **binding})
