from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_22A_PRIVATE_REPOSITORY_INDEPENDENT_RUNTIME_CAPSULE_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evidence_binds_reports_capsules_and_claim() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["automatic_pass"] is True
    assert evidence["status"] == "PASS_PRIVATE_U7_22A_REPOSITORY_INDEPENDENT_RUNTIME_CAPSULE"
    reports = evidence["formal_execution"]["reports"]
    assert reports[0]["sha256"] == reports[1]["sha256"]
    for row in reports:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert _sha256(path) == row["sha256"]

    for capsule in evidence["capsules"]:
        root = ROOT / capsule["path"]
        files = [path for path in root.rglob("*") if path.is_file()]
        assert len(files) == capsule["materialized_file_count"]
        assert sum(path.stat().st_size for path in files) == capsule["logical_bytes"]
        for artifact in evidence["artifact_identity"].values():
            path = root / artifact["path"]
            assert path.stat().st_size == artifact["bytes"]
            assert _sha256(path) == artifact["sha256"]

    assert all(evidence["gates"].values())
    assert evidence["claim_ceiling"]["public_release"] is False
    assert evidence["claim_ceiling"]["redistribution_authorized"] is False
    assert evidence["claim_ceiling"]["calibrated_stock_response"] is False


def test_evidence_binds_committed_sources() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for relative, binding in evidence["source_bindings"].items():
        assert_historical_evidence_binding(ROOT, {"path": relative, **binding})
