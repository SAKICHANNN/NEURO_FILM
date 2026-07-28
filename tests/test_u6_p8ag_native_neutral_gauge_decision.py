from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ag_decision_matches_formal_report() -> None:
    decision = json.loads(
        (
            ROOT
            / "configs/u6_p8ag_native_neutral_gauge_decision_v1.json"
        ).read_text()
    )
    assert _sha256(ROOT / decision["contract"]) == decision[
        "contract_sha256"
    ]
    report_path = ROOT / decision["formal_report"]
    assert _sha256(report_path) == decision["formal_report_sha256"]
    report = json.loads(report_path.read_text())
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    assert report["knot_counts"] == decision["result"]["knot_counts"]
    assert report["oracle"][
        "maximum_absolute_error_vs_python_float32"
    ] == 0.0
    assert report["oracle"]["in_place_byte_exact"]
    assert report["oracle"]["invalid_input_output_unchanged"]
    assert report["double_counting_control"][
        "physical_parameters_reused_as_ao6_parameters"
    ] is False
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AH")
