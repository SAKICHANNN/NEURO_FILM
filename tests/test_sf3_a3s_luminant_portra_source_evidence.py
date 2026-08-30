from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3s_evidence_binds_formal_report() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = root / "docs/evidence/SF3_A3S_LUMINANT_PORTRA_SOURCE_RESULT.json"
    payload = evidence_path.read_bytes()
    report = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == (
        "31c77ec2623798565cc042c89e099c4d5ecd29304cc07554a23e328f6651e781"
    )
    assert len(payload) == 10235
    assert report["decision"] == (
        "FAIL_CLOSED_LUMINANT_PORTRA_SINGLE_SOURCE_NUISANCE_AND_PAIRING_GAP"
    )
    assert report["stable_evidence_id"] == (
        "602403bf4b14f64ff04ff38dba3aaf0630d16c6d44bf6fade6322bd96bd94460"
    )
    assert report["inventory"]["count"] == 191
    assert report["inventory"]["label_counts"] == {
        "Kodak Portra 400": 152,
        "Kodak Portra 400 +2": 19,
        "Portra 400": 20,
    }
    assert report["head_sample"]["count"] == 24
    assert report["head_sample"]["total_bytes"] == 124845733
    assert all(report["audit_gates"].values())
    assert not report["admission_gates"]["independent_author_count"]
    assert not report["admission_gates"]["explicit_roll_group_count"]
    assert not report["admission_gates"]["same_scene_neutral_film_pair_count"]
    assert report["operation_counts"]["image_get_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0
    assert report["operation_counts"]["fit_calls"] == 0
    assert report["operation_counts"]["render_calls"] == 0
    assert report["operation_counts"]["score_calls"] == 0
