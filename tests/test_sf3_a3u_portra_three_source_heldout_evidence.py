from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3u_evidence_binds_fail_closed_formal_result() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/evidence/SF3_A3U_PORTRA_THREE_SOURCE_HELDOUT_RESULT.json"
    payload = path.read_bytes()
    evidence = json.loads(payload)

    assert len(payload) == 3808
    assert hashlib.sha256(payload).hexdigest() == (
        "01702c2fa9daba8c1c917b37dac586ad9c591f78bd1241822754e5b421b7d22c"
    )
    assert evidence["formal_reports"]["byte_exact"]
    assert evidence["formal_reports"]["forward_sha256"] == (
        "f5e8976737e62fb6f337749ab7e18564a83bd3aa95d10536dbfb843d0e0d463d"
    )
    assert evidence["decision"] == (
        "FAIL_CLOSED_PORTRA_SOURCE_HELDOUT_IDENTIFIABILITY_BEFORE_LUMINANT_READ"
    )
    assert evidence["primary_metrics"]["balanced_accuracy"] == 0.5
    assert evidence["primary_metrics"]["wrong_stock_specificity"] == 1 / 12
    assert evidence["primary_metrics"]["best_nuisance_control"] == "hog_grayscale"
    assert evidence["primary_metrics"]["primary_minus_best_nuisance"] < 0
    assert evidence["gates"]["cross_split_duplicate_gate"]
    assert not evidence["gates"]["primary_balanced_accuracy"]
    assert not evidence["gates"]["wrong_stock_specificity"]
    assert (
        evidence["operation_counts_per_formal_process"][
            "luminant_gallery_page_get_requests"
        ]
        == 0
    )
    assert (
        evidence["operation_counts_per_formal_process"]["luminant_image_body_requests"]
        == 0
    )
    assert evidence["operation_counts_per_formal_process"]["operator_fit_calls"] == 0
