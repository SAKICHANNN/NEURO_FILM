from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8d_cpu_consumer import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p8d_cpu_resource_characterization_v1.json"
)
DECISION = (
    ROOT
    / "configs"
    / "u6_p8d_cpu_resource_characterization_decision_v1.json"
)


def test_p8d_contract_binds_two_repeat_bounded_local_scenarios() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    profile = validate_contract(config)
    assert profile["node"] == "U6.P8B"
    assert [row["scenario_id"] for row in config["scenarios"]] == [
        "preview-1mp",
        "mobile-standard-12mp",
    ]
    assert {row["repeats"] for row in config["scenarios"]} == {2}
    assert config["safety"]["maximum_process_tree_rss_bytes"] == 8 * 2**30


def test_p8d_decision_retains_correctness_but_rejects_resources() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["all_runs_successful"]
    assert decision["all_repeat_identity_exact"]
    assert not decision["performance_target_pass"]
    assert (
        decision["mobile_standard_12mp"][
            "peak_process_tree_rss_bytes_range"
        ][0]
        > 4 * 2**30
    )
    assert decision["next_leaf"].startswith("U6.P8E")
