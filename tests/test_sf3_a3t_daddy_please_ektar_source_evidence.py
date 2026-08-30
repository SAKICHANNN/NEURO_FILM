from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3t_evidence_binds_formal_report() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = root / "docs/evidence/SF3_A3T_DADDY_PLEASE_EKTAR_SOURCE_RESULT.json"
    payload = evidence_path.read_bytes()
    report = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == (
        "52d428ab981a3bda2995e033eb9c838304990d67ae780d769705149daddcf6ca"
    )
    assert len(payload) == 12540
    assert report["decision"] == (
        "FAIL_CLOSED_DADDY_PLEASE_EKTAR_SINGLE_PROJECT_NUISANCE_AND_PAIRING_GAP"
    )
    assert report["stable_evidence_id"] == (
        "e431c1461df99de76c87bce040304da6bc167c86d411c142990208fba30ee099"
    )
    assert report["child_inventory"]["count"] == 1000
    assert report["child_inventory"]["ordered_id_sha256"] == (
        "c6c8e53a827592c0a2d44d577db85e9e4926e3c55f661c8f7405ff5c310efafd"
    )
    assert report["metadata_summary"]["model_count"] == 100
    assert report["metadata_summary"]["minimum_photos_per_model"] == 10
    assert report["metadata_summary"]["maximum_photos_per_model"] == 10
    assert report["sample"]["count"] == 24
    assert report["sample"]["total_content_bytes"] == 2088587
    assert all(report["audit_gates"].values())
    assert report["admission_gates"]["explicit_content_group_count"]
    assert not report["admission_gates"]["independent_author_source_count"]
    assert not report["admission_gates"]["explicit_roll_group_count"]
    assert not report["admission_gates"]["same_source_cross_stock_control_count"]
    assert not report["admission_gates"]["same_scene_neutral_film_pair_count"]
    assert report["operation_counts"]["image_body_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0
    assert report["operation_counts"]["fit_calls"] == 0
    assert report["operation_counts"]["render_calls"] == 0
    assert report["operation_counts"]["score_calls"] == 0
