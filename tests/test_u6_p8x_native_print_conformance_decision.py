from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8x_decision_matches_formal_report() -> None:
    decision_path = (
        ROOT
        / "configs/u6_p8x_native_print_conformance_decision_v1.json"
    )
    decision = json.loads(decision_path.read_text())
    contract_path = ROOT / decision["contract"]
    report_path = ROOT / decision["formal_report"]
    assert _sha256(contract_path) == decision["contract_sha256"]
    assert _sha256(report_path) == decision["formal_report_sha256"]
    report = json.loads(report_path.read_text())
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    assert report["dll_sha256"] == decision["dll_sha256"]
    assert report["independent_build_dll_sha_exact"]
    assert len(report["replays"]) == 2
    for replay in report["replays"]:
        assert replay["status"] == "pass"
        assert replay["same_binary_repeat_byte_exact"]
        assert replay["invalid_input_output_unchanged"]
        assert replay["inplace_matches_out_of_place"]
        assert (
            replay["maximum_absolute_error"]
            <= replay["tolerance"]
        )
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8Y")
