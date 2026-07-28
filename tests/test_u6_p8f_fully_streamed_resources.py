from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8f_fully_streamed import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8f_fully_streamed_resources_v1.json"


def test_p8f_contract_requires_full_streaming_and_two_repeats() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    profile = validate_contract(config)
    assert profile["node"] == "U6.P8B"
    assert config["execution"][
        "physical_and_display_look_row_streaming_required"
    ]
    assert {row["repeats"] for row in config["scenarios"]} == {2}
