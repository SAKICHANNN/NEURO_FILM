from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P251_ACES2065_OPENEXR_WORKING_IMAGE_INGRESS_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p251_evidence_binds_exact_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    formal = evidence["formal"]
    forward = ROOT / formal["forward_report_path"]
    reverse = ROOT / formal["reverse_report_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert len(forward.read_bytes()) == formal["report_bytes_each"]
    assert _sha256(forward) == formal["report_sha256"]


def test_p251_evidence_preserves_private_opt_in_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_ACES2065_OPENEXR_WORKING_IMAGE_INGRESS"
    assert evidence["official_dependency"]["default_project_dependency_added"] is False
    assert evidence["formal"]["network_reads"] == 0
    assert all(evidence["gates"].values())
    assert evidence["consumer_mapping"] is False
