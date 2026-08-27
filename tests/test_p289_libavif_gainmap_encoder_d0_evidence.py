from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P289_LIBAVIF_GAINMAP_ENCODER_D0_RESULT.json"


def test_p289_evidence_matches_exact_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    forward = ROOT / evidence["formal"]["forward_report_path"]
    reverse = ROOT / evidence["formal"]["reverse_report_path"]
    forward_bytes = forward.read_bytes()

    assert evidence["status"] == "PASS_PRIVATE_LIBAVIF_GAINMAP_ENCODER_D0"
    assert forward_bytes == reverse.read_bytes()
    assert hashlib.sha256(forward_bytes).hexdigest() == evidence["formal"][
        "report_sha256"
    ]

    report = json.loads(forward_bytes)
    assert report["status"] == evidence["status"]
    assert report["stable_identity"] == evidence["formal"]["scientific_identity"]
    assert all(report["scientific"]["gates"].values())
    assert report["scientific"]["candidate_avif_sha256"] == evidence["formal"][
        "candidate_avif_sha256"
    ]


def test_p289_claim_ceiling_excludes_product_and_candidate3() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["consumer_mapping"] is False
    assert evidence["candidate_count_after_result"] == "2/3"
    assert "No complete ISO 21496-1" in evidence["claim_ceiling"]
    assert "product admission" in evidence["claim_ceiling"]
