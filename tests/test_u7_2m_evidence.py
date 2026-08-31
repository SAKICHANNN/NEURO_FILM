from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2M_PRODUCT_IMAGE_RECIPE_TRANSACTION_RESULT.json"


def test_u7_2m_evidence_binds_product_pair_transaction() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == "PASS_PRIVATE_U7_2M_PRODUCT_IMAGE_RECIPE_TRANSACTION"
    for path, binding in evidence["bindings"]["files"].items():
        assert_historical_evidence_binding(
            ROOT,
            {"path": path, "sha256": binding["sha256"]},
        )

    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert len(forward.read_bytes()) == reports["bytes_each"]
    assert hashlib.sha256(forward.read_bytes()).hexdigest() == reports["sha256"]
    report = json.loads(forward.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert len(report["gates"]) == 20
    assert all(report["gates"].values())
    assert report["owned_runtime_residue_zero"] is True

    result = evidence["result"]
    assert result["formal_gate_count"] == result["formal_gate_pass_count"] == 20
    assert result["concurrent_returncodes"] == [0, 1]
    assert (
        evidence["retained_boundaries"]["calibrated_stock_or_physical_film_claim"]
        is False
    )
