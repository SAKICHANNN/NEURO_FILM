from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_5c_verified_preview_session_v1.json"


def test_u4_5c_contract_keeps_full_admission_and_300ms_warm_gate() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    gates = contract["gates"]
    assert gates["maximum_warm_lookup_seconds"] == 0.3
    assert gates["required_fresh_session_admissions"] == 4
    assert gates["warm_lookups_per_session"] == 8
    assert gates["require_complete_admission_hash_validation"] is True
    assert gates["require_owned_immutable_preview_bytes"] is True
    assert gates["require_post_admission_disk_independence"] is True
    assert gates["require_new_admission_tamper_rejection"] is True


def test_u4_5c_claim_ceiling_remains_private_and_noncalibrated() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    claim = contract["claim_ceiling"].lower()
    assert "private" in claim
    assert "non-calibrated" in claim
    assert "no render acceleration" in claim
    assert "no" in claim and "multi-stock completion" in claim
