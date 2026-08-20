from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U1_4E_WORKING_IMAGE_ACES2_ADAPTER_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u1_4e_evidence_is_bound_and_passes_only_private_adapter() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS_PRIVATE_WORKING_IMAGE_ACES2_ADAPTER"
    assert all(payload["gates"].values())
    assert payload["metrics"]["maximum_adapter_direct_absolute_error"] == 0.0
    assert payload["metrics"]["maximum_source_scalar_packed_absolute_error"] == 0.0
    assert payload["execution"]["all_four_report_sha256_exact"] is True
    for key in ("contract", "implementation", "runner", "config"):
        path = ROOT / payload["bindings"][f"{key}_path"]
        assert _sha256(path) == payload["bindings"][f"{key}_sha256"]
    report_path = ROOT / payload["bindings"]["formal_report_path"]
    assert _sha256(report_path) == payload["bindings"]["formal_report_sha256"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stable_evidence_id"] == payload["bindings"]["stable_evidence_id"]
    assert "no ACES certification" in payload["claim_ceiling"]
    assert "no ACES certification" in report["claim_ceiling"]
