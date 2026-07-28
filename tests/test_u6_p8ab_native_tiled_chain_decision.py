from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ab_decision_matches_formal_report() -> None:
    decision = json.loads(
        (
            ROOT
            / "configs/u6_p8ab_native_tiled_chain_decision_v1.json"
        ).read_text()
    )
    assert _sha256(ROOT / decision["contract"]) == decision[
        "contract_sha256"
    ]
    report_path = ROOT / decision["formal_report"]
    assert _sha256(report_path) == decision["formal_report_sha256"]
    report = json.loads(report_path.read_text())
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    assert report["total_halo"] == decision["result"]["total_halo"]
    assert len(report["rows"]) == 12
    assert all(
        row["byte_exact"]
        and row["maximum_absolute_error"] == 0.0
        and row["output_sha256"] == report["full_output_sha256"]
        for row in report["rows"]
    )
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AC")
