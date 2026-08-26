from __future__ import annotations

import json
from pathlib import Path


def test_p220_evidence_closes_only_exact_portable_parity() -> None:
    evidence = json.loads(
        Path(
            "docs/evidence/P220_INTERIOR_LOGIT_HDR_PORTABLE_PARITY_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["status"] == "FAIL_CLOSED_PORTABLE_LIBM_REPLAY"
    assert evidence["passed_mechanics"]["maximum_absolute_error_gate"] is True
    assert evidence["passed_mechanics"]["failure_atomicity_all_pass"] is True
    assert (
        evidence["failed_gates"]["maximum_relative_error"]["pass_both_reports"] is False
    )
    assert evidence["failed_gates"]["forward_reverse_scientific_exact"] is False
    assert evidence["producer_science_repeated"] is False
    assert evidence["consumer_mapping"] is False
    assert evidence["product_admission"] is False
