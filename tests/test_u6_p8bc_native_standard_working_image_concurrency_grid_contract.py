from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8bc_native_standard_working_image_concurrency_grid import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/u6_p8bc_native_standard_working_image_concurrency_grid_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8bc_native_standard_working_image_concurrency_grid_decision_v1.json"
)


def test_p8bc_grid_is_bounded_and_preserves_failed_parent_gate() -> None:
    config = json.loads(CONTRACT.read_text())
    parent, base = validate_contract(config)
    assert config["candidate_workers"] == [1, 2, 3]
    assert config["repeats_per_candidate"] == 2
    assert parent["result"]["memory_pass"] is False
    assert base["gates"]["maximum_peak_process_tree_rss_bytes"] == (
        384 * 1024 * 1024
    )
    assert config["candidate_only_policy_override"]
    assert not config["production_package_changed"]


def test_p8bc_decision_selects_no_policy_and_preserves_gate() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["result"]["status"] == (
        "closed-no-eligible-concurrency"
    )
    assert decision["result"]["selected_workers"] is None
    assert decision["result"]["all_candidates_memory_fail"]
    assert decision["result"]["memory_gate_bytes"] == 384 * 1024 * 1024
    assert not decision["production_package_changed"]
    assert decision["next_leaf"].startswith("U6.P8BD")
