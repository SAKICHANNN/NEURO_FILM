from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8d_cpu_consumer import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p8d_cpu_resource_characterization_v1.json"
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
