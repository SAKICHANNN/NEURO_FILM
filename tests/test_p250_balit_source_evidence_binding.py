from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P250_BALIT_PAIRED_RETOUCH_SOURCE_ELIGIBILITY_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p250_evidence_binds_exact_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    formal = evidence["formal"]
    forward = ROOT / formal["forward_report_path"]
    reverse = ROOT / formal["reverse_report_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert len(forward.read_bytes()) == formal["report_bytes_each"]
    assert _sha256(forward) == formal["report_sha256"]


def test_p250_evidence_preserves_source_and_claim_boundaries() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"].startswith("NOT_READY_")
    assert evidence["formal"]["dataset_file_requests"] == 0
    assert evidence["formal"]["image_or_thumbnail_requests"] == 0
    assert evidence["formal"]["authenticated_requests"] == 0
    assert evidence["gates"]["paired_observation"]
    assert not evidence["gates"]["anonymous_public_payload"]
    assert not evidence["gates"]["commercial_rights"]
    assert not evidence["consumer_mapping"]
