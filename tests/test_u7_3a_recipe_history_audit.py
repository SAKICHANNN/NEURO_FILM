from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u7_3a_desktop_recipe_history import run_audit

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_three_recipe_audit_passes_exactly() -> None:
    report = run_audit()
    scientific = report["scientific"]
    assert scientific["status"] == "PASS_PRIVATE_DESKTOP_RECIPE_HISTORY_CORE"
    assert scientific["forward_catalog_sha256"] == scientific["reverse_catalog_sha256"]
    assert scientific["catalog"]["counts"] == {
        "discovered": 3,
        "valid": 3,
        "invalid": 0,
    }
    assert all(scientific["gates"].values())


def test_tracked_evidence_preserves_read_only_claim_ceiling() -> None:
    evidence = json.loads(
        (ROOT / "docs/evidence/U7_3A_DESKTOP_RECIPE_HISTORY_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "PASS_PRIVATE_DESKTOP_RECIPE_HISTORY_CORE"
    assert evidence["formal_execution"]["reports_byte_exact"] is True
    assert evidence["contract_facts"]["input_output_file_reads"] == 0
    assert evidence["contract_facts"]["product_api_writes"] == 0
    assert (
        evidence["relationship_to_u7_2c"]["u7_2c_proxy_separation_failure_unchanged"]
        is True
    )
    assert "complete desktop GUI" in evidence["decision"]["does_not_open"]
