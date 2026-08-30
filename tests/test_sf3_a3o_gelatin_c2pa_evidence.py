from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3o_formal_evidence_is_bound_and_zero_pixel() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = (
        root / "docs/evidence/SF3_A3O_GELATIN_C2PA_FILM_SCAN_SOURCE_RESULT.json"
    )
    payload = evidence_path.read_bytes()
    evidence = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == (
        "502d7f6ff3184bc7a8d97ecc8d2c7d6ad430d6d78b3794f5e6ef36ec1ba2f39d"
    )
    assert evidence["decision"] == "FAIL_CLOSED_C2PA_SCAN_RIGHTS_OR_TARGET_ASSET_GAP"
    assert evidence["stable_evidence_id"] == (
        "fe28bac3fd8c6f1bbe5875523b0cc711ba348fa949f8a546de145f50e3af0dfd"
    )
    assert evidence["admission"]["observed_target_stock_ids"] == [
        "kodak_portra_400"
    ]
    assert evidence["admission"]["cawg_training_and_mining_not_allowed"]
    assert not evidence["gates"]["complete_target_stock_coverage_present"]
    assert not evidence["gates"][
        "public_assets_manifest_groups_and_fitting_rights_present"
    ]
    assert not evidence["gates"]["training_mining_authorized"]
    assert all(
        count == 0
        for operation, count in evidence["operation_counts"].items()
        if operation != "html_requests"
    )
