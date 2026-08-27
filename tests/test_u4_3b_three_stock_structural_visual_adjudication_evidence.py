from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_3b_three_stock_structural_visual_adjudication_v1.json"
MATERIAL_ROOT = (
    ROOT / "outputs/eval/u4_3b_three_stock_structural_visual_adjudication_v1"
)
MATERIAL_REPORT = MATERIAL_ROOT / "material_report.json"
EVIDENCE = (
    ROOT / "docs/evidence/U4_3B_THREE_STOCK_STRUCTURAL_VISUAL_ADJUDICATION_RESULT.json"
)


def test_u4_3b_evidence_binds_review_material_and_every_sheet() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    material = json.loads(MATERIAL_REPORT.read_text(encoding="utf-8"))
    assert (
        evidence["config"]["sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    )
    assert (
        evidence["material_report"]["sha256"]
        == hashlib.sha256(MATERIAL_REPORT.read_bytes()).hexdigest()
    )
    assert evidence["material_report"]["bytes"] == MATERIAL_REPORT.stat().st_size
    assert (
        evidence["material_report"]["scientific_identity"]
        == material["scientific_identity"]
    )
    for row in material["scientific_payload"]["sheets"]:
        path = MATERIAL_ROOT / row["sheet"]["relative_path"]
        assert path.stat().st_size == row["sheet"]["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sheet"]["sha256"]


def test_u4_3b_evidence_adjudicates_exact_frozen_pair_set_without_promotion() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    material = json.loads(MATERIAL_REPORT.read_text(encoding="utf-8"))
    material_pairs = {
        (row["source_id"], row["arm_id"])
        for row in material["scientific_payload"]["sheets"]
    }
    adjudicated_pairs = {
        (row["source_id"], row["arm_id"]) for row in evidence["adjudications"]
    }
    assert adjudicated_pairs == material_pairs
    assert len(adjudicated_pairs) == 11
    assert evidence["decision_counts"] == {
        "PASS_NO_CONFIRMED_SEVERE_ARTIFACT": 11,
        "FAIL_CONFIRMED_SEVERE_ARTIFACT": 0,
        "REVIEW_UNRESOLVED": 0,
    }
    assert all(
        row["decision"] == "PASS_NO_CONFIRMED_SEVERE_ARTIFACT"
        for row in evidence["adjudications"]
    )
    assert evidence["review_protocol"]["post_review_substitution_count"] == 0
    assert evidence["review_protocol"]["aggregate_scalar_score_present"] is False
    assert "not population preference" in evidence["claim_ceiling"]
