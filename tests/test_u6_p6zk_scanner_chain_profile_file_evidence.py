from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U6_P6ZK_SCANNER_CHAIN_PROFILE_FILE_RESULT.json"
FORWARD = ROOT / "outputs/eval/u6_p6zk_scanner_chain_profile_file_v1/formal_forward.json"
REVERSE = ROOT / "outputs/eval/u6_p6zk_scanner_chain_profile_file_v1/formal_reverse.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u6_p6zk_evidence_binds_exact_two_report_pass() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_BOUNDED_SCANNER_CHAIN_PROFILE_FILE"
    assert all(evidence["gate_results"].values())
    assert FORWARD.read_bytes() == REVERSE.read_bytes()
    assert FORWARD.stat().st_size == evidence["bindings"]["formal_report_bytes"]
    assert _sha256(FORWARD) == evidence["bindings"]["formal_report_sha256"]
    report = json.loads(FORWARD.read_text(encoding="utf-8"))
    assert report["automatic_pass"] is True
    assert report["stable_evidence_id"] == evidence["bindings"][
        "stable_evidence_id"
    ]
    assert report["file_sha256"] == evidence["bindings"]["file_sha256"]
    assert report["output_sha256"] == evidence["bindings"]["output_sha256"]
