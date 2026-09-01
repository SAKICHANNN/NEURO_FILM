from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_1a_three_stock_gold_severe_adjudication_v1.json"
CONTRACT = (
    ROOT
    / "docs/planning/U4_1A_THREE_STOCK_GOLD_SEVERE_ADJUDICATION_CONTRACT.md"
)
EVIDENCE = (
    ROOT
    / "docs/evidence/U4_1A_THREE_STOCK_GOLD_SEVERE_ADJUDICATION_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u4_1a_evidence_binds_exact_formal_execution_and_review_material() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    formal = evidence["formal_reports"]
    forward = ROOT / formal["forward_path"]
    reverse = ROOT / formal["reverse_path"]
    forward_root = ROOT / formal["forward_media_root"]
    reverse_root = ROOT / formal["reverse_media_root"]
    report = json.loads(forward.read_text(encoding="utf-8"))

    assert evidence["config"]["sha256"] == _sha256(CONFIG)
    assert evidence["contract"]["sha256"] == _sha256(CONTRACT)
    assert forward.read_bytes() == reverse.read_bytes()
    assert forward.stat().st_size == formal["bytes_each"]
    assert _sha256(forward) == formal["sha256"]
    assert report["scientific_identity"] == formal["scientific_identity"]
    assert report["scientific_payload"]["execution"]["source_commit"] == evidence[
        "formal_execution_commit"
    ]
    assert report["status"] == "PASS_OPEN_AUTONOMOUS_VISUAL_REVIEW"

    sheets = report["scientific_payload"]["review_sheets"]
    assert len(sheets) == 27
    for row in sheets:
        forward_sheet = forward_root / row["relative_path"]
        reverse_sheet = reverse_root / row["relative_path"]
        assert forward_sheet.read_bytes() == reverse_sheet.read_bytes()
        assert forward_sheet.stat().st_size == row["bytes"]
        assert _sha256(forward_sheet) == row["sha256"]

    rows = report["scientific_payload"]["rows"]
    assert len(rows) == 27
    assert len({row["output"]["sha256"] for row in rows}) == 27
    assert len({row["recipe"]["semantic_identity"] for row in rows}) == 27
    assert all(row["strict_replay_byte_exact"] for row in rows)
    assert max(row["new_exact_boundary_fraction"] for row in rows) == 0.0


def test_u4_1a_evidence_records_all_81_frozen_visual_verdicts() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report = json.loads(
        (ROOT / evidence["formal_reports"]["forward_path"]).read_text(
            encoding="utf-8"
        )
    )
    protocol = evidence["review_protocol"]
    expected_sources = {"01", "05", "08", "09", "11", "18", "21", "29", "FS_FACE_01"}
    expected_styles = {"velvia_50", "portra_400", "ektar_100"}
    expected_orders = protocol["arm_orders"]

    assert protocol["verdict_count"] == 81
    assert protocol["automatic_metrics_used_as_visual_substitute"] is False
    adjudications = evidence["source_adjudications"]
    assert {row["source_id"] for row in adjudications} == expected_sources
    assert set(report["scientific_payload"]["input_manifest"]["selected_gold_ids"]) == expected_sources

    verdicts: list[str] = []
    for row in adjudications:
        assert len(row["pass_verdicts"]) == 3
        for pass_index, pass_verdicts in enumerate(row["pass_verdicts"]):
            assert list(pass_verdicts) == expected_orders[pass_index]
            assert set(pass_verdicts) == expected_styles
            verdicts.extend(pass_verdicts.values())

    assert verdicts == ["PASS_NO_CONFIRMED_SEVERE_ARTIFACT"] * 81
    assert evidence["decision_counts"] == {
        "PASS_NO_CONFIRMED_SEVERE_ARTIFACT": 81,
        "FAIL_CONFIRMED_SEVERE_ARTIFACT": 0,
        "REVIEW_UNRESOLVED": 0,
    }
    assert evidence["retained_boundaries"] == {
        "filmset_commercial_rights_established": False,
        "aesthetic_preference_established": False,
        "target_film_closeness_established": False,
        "calibrated_or_physical_stock_response_established": False,
        "universal_safety_established": False,
        "public_release_or_product_promotion_opened": False,
    }
    assert "not calibrated or physical film response" in evidence["claim_ceiling"]
