from __future__ import annotations

from pathlib import Path

from scripts.audit_u7_2p_product_cli_research_dependency_isolation import (
    CONFIG,
    build_report,
)


def test_u7_2p_report_passes_all_frozen_dependency_isolation_gates(
    tmp_path: Path,
) -> None:
    report = build_report(CONFIG, "forward")
    assert report["status"] == "PASS"
    assert len(report["gates"]) == 12
    assert all(report["gates"].values())
    assert len(report["comparisons"]) == 9
    assert report["product_process_count"] > 0
    assert report["claim_ceiling"].startswith("Private unselected research-import")
