from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/planning/U4_3C_THREE_STOCK_FACE_SEVERE_REVIEW_CONTRACT.md"
CONFIG = ROOT / "configs/u4_3c_three_stock_face_severe_review_v1.json"
EVIDENCE = ROOT / "docs/evidence/U4_3C_THREE_STOCK_FACE_SEVERE_REVIEW_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u4_3c_evidence_binds_contract_config_and_formal_replay() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    forward = ROOT / evidence["formal_reports"]["forward_path"]
    reverse = ROOT / evidence["formal_reports"]["reverse_path"]
    report = json.loads(forward.read_text(encoding="utf-8"))

    assert evidence["contract"]["sha256"] == _sha256(CONTRACT)
    assert evidence["config"]["sha256"] == _sha256(CONFIG)
    assert forward.read_bytes() == reverse.read_bytes()
    assert forward.stat().st_size == evidence["formal_reports"]["bytes_each"]
    assert _sha256(forward) == evidence["formal_reports"]["sha256"]
    assert (
        report["scientific_identity"]
        == evidence["formal_reports"]["scientific_identity"]
    )
    assert report["scientific_payload"]["automatic_gates"] == evidence[
        "automatic_gates"
    ]


def test_u4_3c_evidence_adjudicates_exact_three_stock_face_set_without_promotion() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    forward = ROOT / evidence["formal_reports"]["forward_path"]
    report = json.loads(forward.read_text(encoding="utf-8"))

    expected = {
        (row["film_stock_id"], row["style_id"])
        for row in report["scientific_payload"]["rows"]
    }
    adjudicated = {
        (row["film_stock_id"], row["style_id"])
        for row in evidence["adjudications"]
    }
    assert adjudicated == expected == {
        ("fujifilm_velvia_50", "velvia_50"),
        ("kodak_portra_400", "portra_400"),
        ("kodak_ektar_100", "ektar_100"),
    }
    assert all(
        row["decision"] == "PASS_NO_CONFIRMED_SEVERE_ARTIFACT"
        for row in evidence["adjudications"]
    )
    assert evidence["decision_counts"] == {
        "PASS_NO_CONFIRMED_SEVERE_ARTIFACT": 3,
        "FAIL_CONFIRMED_SEVERE_ARTIFACT": 0,
        "REVIEW_UNRESOLVED": 0,
    }
    assert evidence["retained_boundaries"] == {
        "ao6_included": False,
        "ao6_prior_population_veto_unchanged": True,
        "portra_ektar_controlled_stock_evidence_gap_unchanged": True,
        "candidate_or_profile_changed": False,
    }
    assert evidence["review_protocol"]["original_file_reopened"] is False
    assert "not original-file ingress" in evidence["claim_ceiling"]
    assert "not original file ingress" in report["scientific_payload"]["claim_ceiling"]
