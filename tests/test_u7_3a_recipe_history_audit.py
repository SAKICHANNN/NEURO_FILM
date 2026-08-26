from __future__ import annotations

from scripts.audit_u7_3a_desktop_recipe_history import run_audit


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
