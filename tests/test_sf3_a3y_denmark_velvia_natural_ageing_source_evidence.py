from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3y_evidence_binds_formal_report() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = (
        root / "docs/evidence/SF3_A3Y_DENMARK_VELVIA_NATURAL_AGEING_SOURCE_RESULT.json"
    )
    payload = evidence_path.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        "82d16980335c04a6b220552410d0c0c7e88527ab7d5df03a3d4e1b65e5ab11d2"
    )
    assert len(payload) == 4319
    assert (
        report["decision"]
        == "FAIL_CLOSED_DENMARK_VELVIA_PUBLIC_DATA_RIGHTS_MANIFEST_AND_ROLE_GAP"
    )
    assert report["source_identity_sha256"] == (
        "14afd158d323a9df445edac719c9398cf4c11dbf6ee96ccd4fe28c7d558ff5df"
    )
    assert report["stable_evidence_id"] == (
        "07051bac779de1c7a81e5627619be3ccd7626c0bc37c391ae71b71d94bba4e9e"
    )
    assert all(report["audit_gates"].values())
    assert not any(report["admission_gates"].values())
    assert report["operation_counts"]["spreadsheet_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0
