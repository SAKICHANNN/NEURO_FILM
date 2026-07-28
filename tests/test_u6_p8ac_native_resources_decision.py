from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ac_decision_matches_formal_report() -> None:
    decision = json.loads(
        (
            ROOT
            / "configs/u6_p8ac_native_resources_decision_v1.json"
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
    assert report["worker_elapsed_seconds"] == decision["result"][
        "worker_elapsed_seconds"
    ]
    assert max(report["peak_process_tree_rss_bytes"]) <= report[
        "memory_gate_bytes"
    ]
    assert max(report["worker_elapsed_seconds"]) <= report[
        "elapsed_gate_seconds"
    ]
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AD")
