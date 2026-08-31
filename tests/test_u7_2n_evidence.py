from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2N_PRODUCT_AUXILIARY_OUTPUT_TRANSACTION_RESULT.json"


def test_u7_2n_evidence_binds_complete_product_bundle_transaction() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert (
        evidence["decision"]
        == "PASS_PRIVATE_U7_2N_PRODUCT_AUXILIARY_OUTPUT_TRANSACTION"
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
    assert len(payload) == reports["bytes_each"] == 4493
    assert hashlib.sha256(payload).hexdigest() == reports["sha256"]

    report = json.loads(payload)
    assert report["status"] == "PASS"
    assert report["implementation_commit"] == "996faf4c"
    assert len(report["gates"]) == 41
    assert all(report["gates"].values())
    assert report["owned_runtime_residue_zero"] is True
    assert report["concurrent_returncodes"] == [0, 1]
    assert report["full_bundle"] == evidence["full_effects_bundle"]

    result = evidence["result"]
    assert result["formal_gate_count"] == result["formal_gate_pass_count"] == 41
    assert result["all_stage_content_and_identity_mutations_rejected"] is True
    assert result["all_foreign_replacements_and_additions_preserved"] is True
    assert (
        evidence["retained_boundaries"]["calibrated_stock_or_physical_film_claim"]
        is False
    )
    assert (
        evidence["retained_boundaries"][
            "adjacent_transaction_wrapper_expansion_authorized"
        ]
        is False
    )
