from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P304_RAWHDR_PAIRED_RAW_SOURCE_READINESS_RESULT.json"


def test_p304_evidence_closes_before_payload_access() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == (
        "FAIL_CLOSED_RAWHDR_SOURCE_RIGHTS_OR_MANIFEST_GAP_NOT_SCIENTIFIC_RESULT"
    )
    assert all(evidence["passed_gates"].values())
    assert not any(evidence["failed_gates"].values())
    assert evidence["access_accounting"]["archive_or_dataset_requests"] == 0
    assert evidence["access_accounting"]["pixel_decodes"] == 0
    assert evidence["source_facts"]["dataset_license"] is None
    assert evidence["formal_reports"]["byte_exact"]
    assert evidence["formal_reports"]["sha256_each"] == (
        "91f81139ff1a83ac190b36f0d1383320d4577ebcec59899a29dc67814ec07175"
    )
