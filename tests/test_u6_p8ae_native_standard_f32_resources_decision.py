from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ae_decision_matches_formal_report() -> None:
    decision = json.loads(
        (
            ROOT
            / "configs/"
            "u6_p8ae_native_standard_f32_resources_decision_v1.json"
        ).read_text()
    )
    assert _sha256(ROOT / decision["contract"]) == decision[
        "contract_sha256"
    ]
    report_path = ROOT / decision["formal_report"]
    assert _sha256(report_path) == decision["formal_report_sha256"]
    report = json.loads(report_path.read_text())
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    assert report["pass"]
    assert report["output_identity_exact"]
    assert report["peak_process_tree_rss_bytes"] == decision["result"][
        "peak_process_tree_rss_bytes"
    ]
    assert report["median_rss_ratio_vs_float64_reference"] == decision[
        "result"
    ]["median_rss_ratio_vs_float64_reference"]
    assert report["memory_pass"]
    assert report["ratio_pass"]
    assert report["elapsed_pass"]
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AF")
