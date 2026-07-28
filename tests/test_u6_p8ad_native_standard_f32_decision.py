from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ad_decision_matches_formal_report() -> None:
    decision = json.loads(
        (
            ROOT
            / "configs/u6_p8ad_native_standard_f32_decision_v1.json"
        ).read_text()
    )
    assert _sha256(ROOT / decision["contract"]) == decision[
        "contract_sha256"
    ]
    report_path = ROOT / decision["formal_report"]
    assert _sha256(report_path) == decision["formal_report_sha256"]
    report = json.loads(report_path.read_text())
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    assert report["maximum_absolute_error_vs_float64"] == decision[
        "result"
    ]["maximum_absolute_error_vs_float64"]
    assert report["maximum_absolute_error_vs_float64"] <= report[
        "tolerance"
    ]
    assert all(report["replay"]["failure_atomic"].values())
    assert report["replay"]["final_array_sha256"] == decision["result"][
        "final_array_sha256"
    ]
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AE")
