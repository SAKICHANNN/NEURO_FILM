from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_3a_three_stock_structural_diagnostics_v1.json"
REPORT = ROOT / "outputs/eval/u4_3a_three_stock_structural_diagnostics_v1/report.json"
EVIDENCE = (
    ROOT / "docs/evidence/U4_3A_THREE_STOCK_STRUCTURAL_DIAGNOSTICS_RESULT.json"
)


def test_u4_3a_evidence_binds_exact_formal_report() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert evidence["config"]["sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert evidence["formal_report"]["sha256"] == hashlib.sha256(
        REPORT.read_bytes()
    ).hexdigest()
    assert evidence["formal_report"]["bytes"] == REPORT.stat().st_size
    assert evidence["formal_report"]["scientific_identity"] == report[
        "scientific_identity"
    ]
    assert evidence["status"] == report["status"]


def test_u4_3a_evidence_preserves_diagnostic_only_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    payload = report["scientific_payload"]

    assert len(payload["rows"]) == evidence["coverage"]["rows"] == 64
    assert len(payload["review_queues"]) == evidence["coverage"][
        "independent_review_queues"
    ] == 7
    assert payload["aggregate_scalar_score_present"] is False
    assert payload["automatic_veto_present"] is False
    assert report["render_calls"] == evidence["coverage"]["render_calls"] == 0
    assert report["network_reads"] == evidence["coverage"]["network_reads"] == 0
    assert all(evidence["gate_results"].values())
    assert "not an automatic severe veto" in evidence["claim_ceiling"]

    rows_by_arm: dict[str, list[dict[str, object]]] = {}
    for row in payload["rows"]:
        rows_by_arm.setdefault(row["arm_id"], []).append(row)
    for arm_id, metric_summaries in evidence["arm_summary"].items():
        for summary_name, expected in metric_summaries.items():
            metric_name = summary_name.removesuffix("_min_median_max")
            values = [row["metrics"][metric_name] for row in rows_by_arm[arm_id]]
            actual = [min(values), statistics.median(values), max(values)]
            assert actual == pytest.approx(expected, abs=1e-15)
