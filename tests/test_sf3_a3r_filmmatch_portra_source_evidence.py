from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3r_evidence_binds_formal_report() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = root / "docs/evidence/SF3_A3R_FILMMATCH_PORTRA_SOURCE_RESULT.json"
    payload = evidence_path.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == "ccefea099be1a3cac22bb4a5c94754a217d8b06fc827a770f47cdb1b70a40089"
    assert len(payload) == 4062
    assert report["decision"] == "FAIL_CLOSED_FILMMATCH_PORTRA_DATASET_RIGHTS_MANIFEST_OR_GROUP_GAP"
    assert report["source_identity_sha256"] == "4108053a73772dca6b3e0a2f2ce0b1126b4f3ecf06b1270bd507cb9880bfe5ec"
    assert report["stable_evidence_id"] == "7eb07b08099c8897cea883064df35ae2daa689018a06f132ea527f7af0fee076"
    assert report["gates"]["controlled_portra_method_facts_present"]
    assert not report["gates"]["public_portra_specific_observation_payload_present"]
    assert report["operation_counts"]["image_requests"] == 0
    assert report["operation_counts"]["drive_body_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0
