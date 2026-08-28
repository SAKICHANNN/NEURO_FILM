from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P319_R1GR_LEAF_MOS_NO_COPY_INTAKE_RESULT.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p319_evidence_binds_reports_and_dependency_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_R1GR_LEAF_MOS_NO_COPY_INTAKE"
    for key in ("config", "audit", "forward_report", "reverse_report"):
        binding = evidence["bindings"][key]
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha(path) == binding["sha256"]
    result = evidence["formal_result"]
    assert result["reports_byte_exact"] is True
    assert result["gates_passed"] == result["gates_total"] == 9
    assert result["callable_and_dependency_blobs_exact"] is True
    assert result["truncate_one_byte_rejected_each_source"] is True
    assert evidence["integrity"]["consumer_core_copied"] is False
    assert evidence["integrity"]["producer_runner_executed"] is False
    assert evidence["integrity"]["producer_reference_decoder_executed"] is False
    assert evidence["rights_and_product"]["product_mapping"] is False
    assert evidence["rights_and_product"]["candidate_3"] is False
