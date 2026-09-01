from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2R_PRODUCT_EFFECT_ARGUMENT_PREFLIGHT_RESULT.json"


def test_u7_2r_evidence_closes_only_the_signed_seed_execution_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == (
        "FAIL_CLOSED_U7_2R_SIGNED_SEED_EXECUTION_BOUNDARY"
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
    assert len(payload) == reports["bytes_each"] == 6694
    assert hashlib.sha256(payload).hexdigest() == reports["sha256"]

    report = json.loads(payload)
    assert report["status"] == "FAIL_CLOSED"
    assert len(report["gates"]) == 10
    assert sum(report["gates"].values()) == 9
    assert {name for name, passed in report["gates"].items() if not passed} == {
        "valid_numeric_boundaries_execute"
    }
    assert len(report["invalid_rows"]) == 17
    assert all(row["returncode"] == 2 for row in report["invalid_rows"])
    assert all(row["message_exact"] for row in report["invalid_rows"])
    assert all(row["input_unmentioned"] for row in report["invalid_rows"])
    assert all(row["artifacts_absent"] for row in report["invalid_rows"])

    failed_boundaries = [
        row
        for row in report["boundary_rows"]
        if row["returncode"] != 0 or not row["output_exists"]
    ]
    assert failed_boundaries == [
        {
            "option": "--seed",
            "output_exists": False,
            "output_sha256": None,
            "returncode": 1,
            "value": "-2147483648",
        }
    ]
    assert all(row["output_exact"] for row in report["default_look_rows"])
    assert report["full_effects_bundle"]["output_exact_to_u7_2o"] is True
    assert report["full_effects_bundle"]["layers_exact_to_u7_2o"] is True
    assert report["full_effects_bundle"][
        "recipe_semantics_exact_to_reference"
    ] is True
    assert report["full_effects_bundle"][
        "metrics_semantics_exact_to_reference"
    ] is True
    assert all(
        row["blob_exact"] and row["sha256_exact"]
        for row in report["source_locks"].values()
    )

    result = evidence["result"]
    assert result["formal_gate_count"] == 10
    assert result["formal_gate_pass_count"] == 9
    assert result["all_invalid_cases_rejected_predecode"] is True
    assert result["all_invalid_cases_publish_nothing"] is True
    assert result["all_other_valid_boundaries_execute"] is True
    assert evidence["retained_product_change"][
        "signed_int32_seed_range_fully_executable"
    ] is False

    boundaries = evidence["retained_boundaries"]
    assert boundaries["complete_product_effect_argument_preflight_claim"] is False
    assert boundaries["negative_seed_rescue_authorized"] is False
    assert boundaries["calibrated_stock_response_or_physical_film_claim"] is False
    assert boundaries["adjacent_argument_wrapper_expansion_authorized"] is False
