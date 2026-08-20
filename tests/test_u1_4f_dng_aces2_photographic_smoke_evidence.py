from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U1_4F_DNG_ACES2_PHOTOGRAPHIC_SMOKE_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u1_4f_evidence_is_bound_and_claim_limited() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS_PRIVATE_FOUR_DNG_ACES2_PHOTOGRAPHIC_SMOKE"
    assert all(payload["gates"].values())
    assert payload["execution"]["all_four_report_sha256_exact"] is True
    assert payload["execution"]["source_count"] == 4
    assert payload["execution"]["decoded_payloads_retained"] is False
    assert payload["metrics"]["maximum_observed_outside_unit_fraction"] == 0.0
    for key in ("contract", "adapter", "runner", "config"):
        path = ROOT / payload["bindings"][f"{key}_path"]
        assert _sha256(path) == payload["bindings"][f"{key}_sha256"]
    report_path = ROOT / payload["bindings"]["formal_report_path"]
    assert _sha256(report_path) == payload["bindings"]["formal_report_sha256"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stable_evidence_id"] == payload["bindings"]["stable_evidence_id"]
    assert "no vendor parity" in payload["claim_ceiling"]
    assert "no vendor parity" in report["claim_ceiling"]
