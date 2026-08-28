from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/P313_CANON_SRAW_WORKING_IMAGE_COMPATIBILITY_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p313_evidence_binds_exact_successful_replay() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert (
        _sha256(EVIDENCE)
        == "7aaf526207c7fd76d71ad7b1e7f2853e48a1fc8f7cb3359a1feaf27c07bf4e3a"
    )
    assert evidence["status"] == "PASS_PRIVATE_CANON_SRAW_WORKING_IMAGE_COMPATIBILITY"
    replay = evidence["formal_replay"]
    assert replay["reports_byte_exact"]
    assert replay["forward_report_sha256"] == replay["reverse_report_sha256"]
    assert len(evidence["records"]) == 6
    assert all(evidence["gates"].values())
