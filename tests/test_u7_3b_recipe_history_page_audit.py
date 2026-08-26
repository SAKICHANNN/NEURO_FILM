from __future__ import annotations

from scripts.audit_u7_3b_offline_recipe_history_page import run_audit


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
