from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3v_evidence_binds_formal_report() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = (
        root
        / "docs/evidence/SF3_A3V_NORITSU_PORTRA_PHYSICAL_PROFILE_SOURCE_RESULT.json"
    )
    payload = evidence_path.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        "2f7e1f9648be557fd9085464fe3be5cb017573e53cb12065e034a7d1f2ccc43f"
    )
    assert len(payload) == 4813
    assert (
        report["decision"]
        == "FAIL_CLOSED_NORITSU_PORTRA_PRIVATE_PAIR_RIGHTS_AND_GROUP_GAP"
    )
    assert (
        report["source_identity_sha256"]
        == "facbc659bcf5cfda777a0823ef21e3b6e218e3389640943bf8366b341cde998e"
    )
    assert (
        report["stable_evidence_id"]
        == "51d0b34bdf2da4f13236d066210d51d6998a8ea96c18a7584b9b77f438aa9522"
    )
    assert report["gates"]["real_portra_machine_pair_statements_present"]
    assert not report["gates"]["owner_pair_payload_publicly_addressable"]
    assert report["operation_counts"]["profile_body_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0
