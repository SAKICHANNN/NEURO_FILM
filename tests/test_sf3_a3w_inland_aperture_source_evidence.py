from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3w_evidence_binds_formal_report() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = (
        root
        / "docs/evidence/SF3_A3W_INLAND_APERTURE_PORTRA_EKTAR_SOURCE_RESULT.json"
    )
    payload = evidence_path.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        "297dc36b94816eb6f89a429cfca8fb055d1daa349e051a6acc559ad73e0e4d0b"
    )
    assert len(payload) == 8992
    assert report["decision"] == (
        "FAIL_CLOSED_INLAND_APERTURE_SINGLE_SOURCE_GROUP_AND_PAIRING_GAP"
    )
    assert report["manifest"]["sha256"] == (
        "e232545511feb1ecdac64ca6b62da466e43907211f58ac5d9f8faedb704bcdf6"
    )
    assert report["rights"]["sha256"] == (
        "60a4d729cf8f494259c5dc414b8061e21a109656103fb745451c3931cd948863"
    )
    assert report["stable_evidence_id"] == (
        "666dca135e7f191405268ff42c78b4f9820681735f88c2294061e68d6ff03b41"
    )
    assert all(report["audit_gates"].values())
    assert report["manifest"]["stock_counts"] == {
        "ektar_100": 3,
        "portra_400": 4,
    }
    assert not report["admission_gates"]["independent_author_source_count"]
    assert not report["admission_gates"]["sealed_confirmation_group_count"]
    assert report["operation_counts"]["image_body_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0
