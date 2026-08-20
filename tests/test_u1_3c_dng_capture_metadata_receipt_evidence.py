from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U1_3C_DNG_CAPTURE_METADATA_RECEIPT_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u1_3c_evidence_binds_tracked_protocol_and_implementation() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_DNG_CAPTURE_METADATA_RECEIPT"
    assert evidence["gates"]["all"] is True
    assert evidence["metrics"]["raster_decode_calls"] == 0
    for key in ("contract", "implementation", "formal_config", "formal_runner"):
        binding = evidence["bindings"][key]
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]


def test_u1_3c_evidence_preserves_claim_ceiling() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    excluded = " ".join(evidence["claim_ceiling"]["not_established"])
    assert "arbitrary DNG" in excluded
    assert "renderer or WorkingImage integration" in excluded
    assert "film, stock, product or commercial admission" in excluded
    assert evidence["execution_integrity_note"]["preformal_reports_discarded"] is True

