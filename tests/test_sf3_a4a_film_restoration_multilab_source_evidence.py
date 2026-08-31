from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a4a_evidence_binds_formal_result() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = (
        root
        / "docs/evidence/SF3_A4A_FILM_RESTORATION_MULTILAB_SOURCE_RESULT.json"
    )
    payload = evidence_path.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        "9826836ea5bbf690818365793dc7b38761e2441cbfa86d45ccdec844c8d3ceb8"
    )
    assert len(payload) == 6202
    assert report["decision"] == (
        "FAIL_CLOSED_FILM_RESTORATION_MULTILAB_DATA_RIGHTS_MANIFEST_AND_ROLE_GAP"
    )
    assert report["source_identity_sha256"] == (
        "4052ebd9ae7b07f0d4422287489664d4df0c89a64257993b3fba2a38f6f78e4b"
    )
    assert report["stable_evidence_id"] == (
        "e8b62de822b16716ec0930ec895b07d602de15eb9ca4a67203a59bbb2c3eca80"
    )
    assert all(report["audit_gates"].values())
    assert not any(report["admission_gates"].values())
    assert report["source"]["public_article_topology"]["film_stock_identity"] == (
        "unknown"
    )
    assert report["operation_counts"]["source_negative_requests"] == 0
    assert report["operation_counts"]["laboratory_version_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0
