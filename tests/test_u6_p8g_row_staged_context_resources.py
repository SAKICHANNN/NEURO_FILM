from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8g_row_staged_context import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p8g_row_staged_context_resources_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8g_row_staged_context_resources_decision_v1.json"
)


def test_p8g_contract_reuses_parent_measurement_without_pixel_change() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = validate_contract(config)
    assert parent["node"] == "U6.P8F"
    assert (
        config["implementation"]["source_context_density_execution"]
        == "row-staged"
    )
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]


def test_p8g_decision_retains_context_and_opens_lifetime_work() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["all_repeat_identity_exact"]
    assert decision["same_output_identity_as_p8d_and_p8f"]
    assert (
        decision["comparison_to_p8f"][
            "twelve_mp_peak_rss_reduction_fraction_at_mean"
        ]
        > 0.5
    )
    assert not decision["performance_target_pass"]
    assert decision["next_leaf"].startswith("U6.P8H")
