from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_u7_2c_contract_is_frozen_before_execution() -> None:
    contract = json.loads(
        (ROOT / "configs/u7_2c_three_stock_look_amount_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["status"] == "FROZEN_BEFORE_FORMAL_EXECUTION"
    assert contract["implementation_commit"] == "972830be"
    assert contract["amounts"] == [0.0, 0.5, 1.0]
    assert contract["film_stock_ids"] == [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    assert (
        contract["gates"]["minimum_pairwise_full_output_population_median_delta_e76"]
        == 1.0
    )
    assert "calibrated stock response" in contract["forbidden_claims"]


def test_u7_2c_runner_keeps_timing_out_of_stable_payload() -> None:
    source = (ROOT / "scripts/audit_u7_2c_three_stock_look_amount.py").read_text(
        encoding="utf-8"
    )
    stable_block = source.split("stable = {", 1)[1].split("report = {", 1)[0]
    assert "wall_seconds" not in stable_block
    assert "pairwise_full_output_delta_e76" in stable_block
