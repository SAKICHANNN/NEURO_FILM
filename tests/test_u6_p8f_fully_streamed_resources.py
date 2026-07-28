from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8f_fully_streamed import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8f_fully_streamed_resources_v1.json"
DECISION = (
    ROOT / "configs/u6_p8f_fully_streamed_resources_decision_v1.json"
)


def test_p8f_contract_requires_full_streaming_and_two_repeats() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    profile = validate_contract(config)
    assert profile["node"] == "U6.P8B"
    assert config["execution"][
        "physical_and_display_look_row_streaming_required"
    ]
    assert {row["repeats"] for row in config["scenarios"]} == {2}


def test_p8f_decision_retains_exact_topology_but_rejects_target() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["all_repeat_identity_exact"]
    assert decision["same_output_identity_as_p8d"]
    assert (
        decision["comparison_to_p8d"][
            "twelve_mp_worker_time_reduction_fraction_at_median"
        ]
        > 0.4
    )
    assert not decision["performance_target_pass"]
    assert decision["next_leaf"].startswith("U6.P8G")
