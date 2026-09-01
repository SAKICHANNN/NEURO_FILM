from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2O_PRODUCT_LOOK_CLI_ENTRY_RESULT.json"


def test_u7_2o_evidence_binds_first_class_product_look_cli_entry() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == "PASS_PRIVATE_U7_2O_PRODUCT_LOOK_CLI_ENTRY"
    for path, binding in evidence["bindings"]["files"].items():
        assert_historical_evidence_binding(
            ROOT,
            {"path": path, **binding},
        )

    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    payload = forward.read_bytes()
    assert payload == reverse.read_bytes()
    assert len(payload) == reports["bytes_each"] == 7617
    assert hashlib.sha256(payload).hexdigest() == reports["sha256"]

    report = json.loads(payload)
    assert report["status"] == "PASS"
    assert report["contract_commit"] == "417ff63c"
    assert report["implementation_commit"] == "f02ca086"
    assert len(report["gates"]) == 12
    assert all(report["gates"].values())
    assert len(report["comparisons"]) == 9
    assert all(row["image_byte_exact"] for row in report["comparisons"])
    assert all(row["recipe_semantics_exact"] for row in report["comparisons"])
    assert {row["profile_id"] for row in report["comparisons"]} == {
        "safe-rich-product-v1"
    }
    assert {row["recipe_claim_label"] for row in report["comparisons"]} == {
        "film-inspired"
    }
    assert not any(row["recipe_claim_calibrated"] for row in report["comparisons"])
    assert report["full_bundle"] == evidence["full_effects_bundle"]

    result = evidence["result"]
    assert result["formal_gate_count"] == result["formal_gate_pass_count"] == 12
    assert result["comparison_count"] == 9
    assert result["all_conflicts_rejected_predecode"] is True
    assert result["all_non_product_values_rejected"] is True
    assert result["parent_auxiliary_transaction_unchanged"] is True
    assert result["owned_runtime_residue_zero"] is True

    boundaries = evidence["retained_boundaries"]
    assert boundaries["calibrated_stock_response_or_physical_film_claim"] is False
    assert (
        boundaries["new_stock_identification_or_multi_stock_calibration_claim"] is False
    )
    assert boundaries["adjacent_cli_wrapper_expansion_authorized"] is False
