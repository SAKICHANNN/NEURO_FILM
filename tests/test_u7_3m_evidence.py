from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/evidence/U7_3M_RESEARCH_RECIPE_PRODUCT_EXPORT_EXCLUSION_RESULT.json"
)
REPORTS = (
    ROOT
    / "outputs/eval/u7_3m_research_recipe_product_export_exclusion_v1/formal_forward.json",
    ROOT
    / "outputs/eval/u7_3m_research_recipe_product_export_exclusion_v1/formal_reverse.json",
)


def test_u7_3m_evidence_binds_exact_passing_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    payloads = [path.read_bytes() for path in REPORTS]
    assert payloads[0] == payloads[1]
    assert len(payloads[0]) == evidence["formal_reports"]["bytes_each"]
    assert hashlib.sha256(payloads[0]).hexdigest() == evidence["formal_reports"][
        "sha256_each"
    ]
    report = json.loads(payloads[0])
    assert report["status"] == evidence["status"]
    assert report["stable_identity"] == evidence["formal_reports"][
        "stable_identity"
    ]
    assert report["report_identity"] == evidence["formal_reports"][
        "report_identity"
    ]
    assert all(report["scientific"]["gates"].values())
    assert report["scientific"]["controls"]["research_request_count"] == 0
    assert report["scientific"]["controls"]["mixed_request_styles"] == [
        "ektar_100"
    ]
    assert evidence["excluded_execution"]["science_or_gate_change"] is False
    assert evidence["excluded_execution"]["excluded_reports_retained_as_formal"] is False

