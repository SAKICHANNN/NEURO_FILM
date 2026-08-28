from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/P314_CANON_SRAW_PRODUCT_CHAIN_COMPATIBILITY_RESULT.json"
)


def test_p314_evidence_binds_exact_six_file_product_replay() -> None:
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == (
        "e2b097ebb5a2c3f91e891c72c52e1d70ecc874332ecadcb5be15d727ca53abb1"
    )
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == ("PASS_PRIVATE_CANON_SRAW_PRODUCT_CHAIN_COMPATIBILITY")
    replay = evidence["formal_replay"]
    assert replay["reports_byte_exact"]
    assert replay["forward_report_sha256"] == replay["reverse_report_sha256"]
    assert len(evidence["records"]) == 6
    assert all(evidence["gates"].values())
    assert all(
        row["output_sha256"] == row["replay_sha256"] for row in evidence["records"]
    )
    assert evidence["candidate_count"] == "2/3"
