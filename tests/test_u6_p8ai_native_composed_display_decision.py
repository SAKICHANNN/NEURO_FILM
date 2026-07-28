from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ai_decision_matches_formal_report() -> None:
    decision = json.loads(
        (
            ROOT
            / "configs/u6_p8ai_native_composed_display_decision_v1.json"
        ).read_text()
    )
    assert _sha256(ROOT / decision["contract"]) == decision[
        "contract_sha256"
    ]
    report_path = ROOT / decision["formal_report"]
    assert _sha256(report_path) == decision["formal_report_sha256"]
    report = json.loads(report_path.read_text())
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    assert len(report["fixture"]["partition_rows"]) == decision["result"][
        "partition_rows"
    ]
    assert all(
        all(
            row[key]
            for key in (
                "physical_byte_exact",
                "gauge_byte_exact",
                "external_base_byte_exact",
                "residual_byte_exact",
            )
        )
        for row in report["fixture"]["partition_rows"]
    )
    assert report["fixture"][
        "maximum_encoded_error_vs_python_residual"
    ] == decision["result"]["maximum_encoded_error_vs_python_residual"]
    assert report["claim_ledger"]["production_path_opened"] is False
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AJ")
