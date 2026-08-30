from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3p_tracked_evidence_binds_formal_result() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = (
        root
        / "docs/evidence/SF3_A3P_FILM2PAINT_REVERSAL_TARGET_SOURCE_RESULT.json"
    )
    payload = evidence_path.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        "66ca26b07ab31a83423b66f32cf32c430379f29ee05efab2aef08f3732c6bbf7"
    )
    assert report["decision"] == (
        "FAIL_CLOSED_FILM2PAINT_DATASET_ASSET_RIGHTS_OR_GROUP_GAP"
    )
    assert report["source_identity_sha256"] == (
        "b3d93c758c4867ce72f1cbfd3a7a1fdd98842881e4168cd32ef2db1d3c0e66ac"
    )
    assert report["stable_evidence_id"] == (
        "9c5f479b87ffd9e3bd6ec4c12535f1b784afd10c639cc09024ca849ec2b4f749"
    )
    assert report["admission"]["dataset_bitstreams"] == []
    assert report["operation_counts"]["film_target_scan_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0
