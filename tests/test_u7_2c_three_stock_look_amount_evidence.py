from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2C_THREE_STOCK_LOOK_AMOUNT_RESULT.json"


def test_u7_2c_evidence_preserves_the_frozen_negative() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["automatic_pass"] is False
    assert evidence["decision"] == (
        "FAIL_CLOSED_LEGACY_THREE_STOCK_PROXIES_ARE_NOT_MATERIALLY_DISTINCT_WITHOUT_RESCUE"
    )
    gates = evidence["mechanism_gates"]
    assert gates["zero_amount_exact_identity"] is True
    assert gates["one_amount_exact_existing_baseline"] is True
    assert gates["half_residual_rmse_strictly_between_zero_and_full"] is True
    assert gates["forward_reverse_exact"] is True
    assert gates["nonzero_output_boundary_fraction"] == 0.0
    pairs = evidence["pairwise_full_output_delta_e76"]
    assert sum(row["passes_frozen_one_delta_e_gate"] for row in pairs) == 1
    assert min(row["median"] for row in pairs) == 0.9279583692550659
    assert "No further legacy-proxy tuning" in evidence["next_leaf"]


def test_u7_2c_report_identities_are_distinct_from_stable_identity() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report_hashes = {row["sha256"] for row in evidence["formal_reports"]}
    assert len(report_hashes) == 2
    assert evidence["stable_identity"] not in report_hashes
