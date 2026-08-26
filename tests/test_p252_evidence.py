from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P252_ACES2065_ACES2_PQ_COMPOSITION_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p252_evidence_binds_exact_corrected_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    execution = evidence["execution"]
    forward = ROOT / execution["forward_report_path"]
    reverse = ROOT / execution["reverse_report_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert len(forward.read_bytes()) == execution["report_bytes_each"]
    assert _sha256(forward) == execution["report_sha256"]
    assert evidence["formal_correction"]["scientific_payload_changed"] is False
    assert evidence["formal_correction"]["algorithm_or_gates_changed"] is False


def test_p252_evidence_preserves_private_boundary_and_legacy_hashes() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_ACES2065_ACES2_PQ_COMPOSITION"
    assert all(evidence["gates"].values())
    assert len(evidence["metrics"]["legacy_output_hashes"]) == 4
    assert evidence["execution"]["candidate_counter_change"] == 0
    assert evidence["execution"]["natural_image_reads"] == 0
    assert evidence["product_mapping"] is False
    assert evidence["capability_mapping"] is False
