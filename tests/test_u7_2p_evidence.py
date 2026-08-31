from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/U7_2P_PRODUCT_CLI_RESEARCH_DEPENDENCY_ISOLATION_RESULT.json"
)


def test_u7_2p_evidence_binds_product_cli_research_dependency_isolation() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert (
        evidence["decision"]
        == "PASS_PRIVATE_U7_2P_PRODUCT_CLI_RESEARCH_DEPENDENCY_ISOLATION"
    )
    for path, binding in evidence["bindings"]["files"].items():
        assert_historical_evidence_binding(
            ROOT,
            {"path": path, "sha256": binding["sha256"]},
        )

    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    payload = forward.read_bytes()
    assert payload == reverse.read_bytes()
    assert len(payload) == reports["bytes_each"] == 8211
    assert hashlib.sha256(payload).hexdigest() == reports["sha256"]

    report = json.loads(payload)
    assert report["status"] == "PASS"
    assert report["contract_commit"] == "96d6d10a"
    assert report["implementation_commit"] == "657b563"
    assert len(report["gates"]) == 12
    assert all(report["gates"].values())
    assert report["product_process_count"] == 31
    assert len(report["comparisons"]) == 9
    assert report["full_bundle"] == evidence["full_effects_bundle"]
    assert report["blocked_import_roots"] == evidence["result"]["blocked_import_roots"]

    result = evidence["result"]
    assert result["formal_gate_count"] == result["formal_gate_pass_count"] == 12
    assert result["product_subprocesses_with_blocked_import_attempts"] == 0
    assert result["all_three_looks_all_amounts_exact_to_u7_2o"] is True
    assert result["analytic_branch_imports_only_when_selected"] is True
    assert result["analytic_branch_still_executes_unblocked"] is True
    assert result["owned_runtime_residue_zero"] is True

    boundaries = evidence["retained_boundaries"]
    assert boundaries["product_dependency_set_or_requirements_changed"] is False
    assert boundaries["calibrated_stock_response_or_physical_film_claim"] is False
    assert (
        boundaries["minimal_install_public_package_installer_or_release_claim"] is False
    )
    assert boundaries["adjacent_cli_wrapper_expansion_authorized"] is False
