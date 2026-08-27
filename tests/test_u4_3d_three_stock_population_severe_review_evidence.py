from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_3d_three_stock_population_severe_review_v1.json"
EVIDENCE = (
    ROOT
    / "docs/evidence/U4_3D_THREE_STOCK_POPULATION_SEVERE_REVIEW_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u4_3d_evidence_binds_config_and_exact_formal_replay() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    forward = ROOT / evidence["formal_reports"]["forward_path"]
    reverse = ROOT / evidence["formal_reports"]["reverse_path"]
    report = json.loads(forward.read_text(encoding="utf-8"))

    assert evidence["config"]["sha256"] == _sha256(CONFIG)
    assert forward.read_bytes() == reverse.read_bytes()
    assert forward.stat().st_size == evidence["formal_reports"]["bytes_each"]
    assert _sha256(forward) == evidence["formal_reports"]["sha256"]
    assert (
        report["scientific_identity"]
        == evidence["formal_reports"]["scientific_identity"]
    )
    assert report["scientific_payload"]["render_calls"] == 0
    assert report["scientific_payload"]["network_reads"] == 0


def test_u4_3d_evidence_adjudicates_exact_population_without_promotion() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report = json.loads(
        (ROOT / evidence["formal_reports"]["forward_path"]).read_text(
            encoding="utf-8"
        )
    )

    expected_arms = {
        "legacy_unpaired_safe_lab_k1_velvia50",
        "legacy_unpaired_safe_lab_k1_portra400",
        "legacy_unpaired_safe_lab_k1_ektar100",
    }
    report_sources = {
        row["source_id"]: {arm["arm_id"] for arm in row["rows"]}
        for row in report["scientific_payload"]["sheets"]
    }
    adjudicated = {
        row["source_id"]: row["decisions"]
        for row in evidence["source_adjudications"]
    }

    assert len(report_sources) == len(adjudicated) == 16
    assert set(report_sources) == set(adjudicated)
    assert all(arms == expected_arms for arms in report_sources.values())
    assert all(
        decisions == ["PASS_NO_CONFIRMED_SEVERE_ARTIFACT"] * 3
        for decisions in adjudicated.values()
    )
    assert evidence["decision_counts"] == {
        "PASS_NO_CONFIRMED_SEVERE_ARTIFACT": 48,
        "FAIL_CONFIRMED_SEVERE_ARTIFACT": 0,
        "REVIEW_UNRESOLVED": 0,
    }
    assert evidence["review_protocol"]["ao6_included"] is False
    assert evidence["retained_boundaries"]["u7_2c_stock_separation_failure_unchanged"]
    assert evidence["retained_boundaries"]["a0n_physical_capture_gate_unchanged"]
    assert evidence["retained_boundaries"]["product_promotion_opened"] is False
    assert "not target-film closeness" in evidence["claim_ceiling"]
