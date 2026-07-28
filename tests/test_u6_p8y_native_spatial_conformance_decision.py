from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8y_decision_matches_formal_report() -> None:
    decision = json.loads(
        (
            ROOT
            / "configs/u6_p8y_native_spatial_conformance_decision_v1.json"
        ).read_text()
    )
    assert _sha256(ROOT / decision["contract"]) == decision[
        "contract_sha256"
    ]
    report_path = ROOT / decision["formal_report"]
    assert _sha256(report_path) == decision["formal_report_sha256"]
    report = json.loads(report_path.read_text())
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    assert report["independent_build_dll_sha_exact"]
    assert len(report["replays"]) == 2
    for replay in report["replays"]:
        assert replay["status"] == "pass"
        assert replay["invalid_input_buffers_unchanged"]
        assert replay["overlap_rejected_before_write"]
        assert [row["stage"] for row in replay["stages"]] == [
            "forward_scatter",
            "development_adjacency",
            "dye_diffusion",
            "scanner_mtf",
        ]
        assert all(
            row["same_binary_repeat_byte_exact"]
            and row["maximum_absolute_error"] <= row["tolerance"]
            for row in replay["stages"]
        )
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8Z")
