from __future__ import annotations

import json
from pathlib import Path

from src.eval.global_frontier import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs/u6_p8w_tile_row_grid_decision_v1.json"
REPORT = ROOT / "outputs/u6_p8w_tile_row_grid_v1/run_a/report.json"


def test_p8w_decision_binds_exact_grid_evidence() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert decision["report_sha256"] == sha256_file(REPORT)
    assert (
        decision["stable_evidence_id"]
        == report["stable_evidence_id"]
    )
    assert decision["all_runs_successful"]
    assert decision["all_output_identity_exact"]
    assert decision["selected_tile_rows"] == 32
    assert report["stable_evidence"]["selected_tile_rows"] == 32


def test_p8w_closes_python_micro_optimization_and_opens_native() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))

    minimum = decision["candidates"]["32"][
        "mean_peak_process_tree_rss_bytes"
    ]
    larger = decision["candidates"]["64"][
        "mean_peak_process_tree_rss_bytes"
    ]
    assert larger - minimum > 8 * 1024 * 1024
    assert decision["python_reference_topology_complete"]
    assert decision["desktop_python_reference_memory_target_pass"]
    assert not decision["mobile_native_memory_target_pass"]
    assert not decision["production_default_changed"]
    assert not decision["native_runtime_opened"]
    assert decision["next_leaf"].startswith("U6.P8X")
