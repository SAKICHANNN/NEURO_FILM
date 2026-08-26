from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u7_3b_offline_recipe_history_page import run_audit

ROOT = Path(__file__).resolve().parents[1]


def test_offline_history_page_formal_audit_passes() -> None:
    report, html = run_audit()
    scientific = report["scientific"]
    assert scientific["status"] == "PASS_PRIVATE_OFFLINE_RECIPE_HISTORY_PAGE"
    assert scientific["html_bytes"] == len(html)
    assert scientific["catalog_counts"] == {
        "discovered": 3,
        "valid": 3,
        "invalid": 0,
    }
    assert all(scientific["gates"].values())


def test_u7_3b_tracked_evidence_binds_formal_and_visual_results() -> None:
    evidence = json.loads(
        (ROOT / "docs/evidence/U7_3B_OFFLINE_RECIPE_HISTORY_PAGE_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    formal = evidence["formal_execution"]
    review = evidence["browser_visual_review"]
    assert evidence["status"] == "PASS_PRIVATE_OFFLINE_RECIPE_HISTORY_PAGE"
    assert formal["reports_byte_exact"] is True
    assert formal["forward_report_sha256"] == formal["reverse_report_sha256"]
    assert formal["html_sha256"] == (
        "efd83a8ad23d2fe02673e0cff5e24f7b6aca9f0e95053936b37cd37f0f66e262"
    )
    assert review["desktop_layout_complete"] is True
    assert review["narrow_layout_complete"] is True
    assert review["owned_edge_profile_directories_removed"] == 10
    assert all(evidence["gates"].values())
    assert evidence["relationship_to_u7_2c"]["u7_2c_proxy_separation_failure_unchanged"]
