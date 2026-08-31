from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3z_evidence_binds_formal_result() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = (
        root
        / "docs/evidence/SF3_A3Z_HERSHENSON_ALLEN_VELVIA_ARCHIVE_SOURCE_RESULT.json"
    )
    payload = evidence_path.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        "90f9b109e0a3d1bf7082f26854df31490bd6f56434218a280f83fd6c99a755ea"
    )
    assert len(payload) == 4081
    assert report["decision"] == (
        "FAIL_CLOSED_HERSHENSON_ALLEN_ITEM_STOCK_RIGHTS_MANIFEST_AND_ROLE_GAP"
    )
    assert report["source_identity_sha256"] == (
        "fc42e1938afa0d6712d2315900cd959d7428ba89367442a3772f9c9b34d9c366"
    )
    assert report["stable_evidence_id"] == (
        "d0434042ec7edd0f4754bf0cde8ab2621e0f5436e435f01b68e6f9228ae6e878"
    )
    assert all(report["audit_gates"].values())
    assert not any(report["admission_gates"].values())
    assert report["operation_counts"]["database_queries"] == 0
    assert report["operation_counts"]["media_body_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0
