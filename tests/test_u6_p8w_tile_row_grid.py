from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8w_tile_row_grid import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8w_tile_row_grid_v1.json"


def test_p8w_contract_freezes_bounded_grid_and_selection() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    profile = validate_contract(config)

    assert profile["node"] == "U6.P8B"
    assert config["scenario"]["tile_rows_candidates"] == [64, 32]
    assert config["scenario"]["repeats_per_candidate"] == 2
    assert (
        config["measurement"][
            "minimum_mean_peak_rss_reduction_fraction"
        ]
        == 0.02
    )
    assert (
        config["measurement"]["largest_tile_within_bytes_of_minimum"]
        == 8 * 1024 * 1024
    )
    assert not config["measurement"]["timing_comparison_valid"]
    assert not config["measurement"]["post_result_retuning_allowed"]
