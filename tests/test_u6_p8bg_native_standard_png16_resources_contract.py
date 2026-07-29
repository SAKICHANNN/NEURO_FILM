from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8bg_native_standard_png16_resources import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p8bg_native_standard_png16_resources_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8bg_native_standard_png16_resources_decision_v1.json"
)


def test_p8bg_contract_is_single_quantization_and_bounded() -> None:
    config = json.loads(CONTRACT.read_text())
    parent, _ = validate_contract(config)
    assert parent["result"]["restart_verification_repeat_exact"]
    assert config["execution"]["single_final_quantization"]
    assert config["execution"]["output_bit_depth"] == 16
    assert config["gates"]["maximum_peak_process_tree_rss_bytes"] == (
        1024 * 1024 * 1024
    )
    assert not config["execution"]["decoder_included"]


def test_p8bg_decision_records_exact_png_and_resource_pass() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["result"]["status"] == "pass"
    assert decision["result"][
        "png_repeat_exact_and_each_receipt_verified"
    ]
    assert decision["result"]["path_bound_transaction_ids_distinct"]
    assert decision["result"]["single_final_quantization"]
    assert decision["result"]["memory_pass"]
    assert decision["result"]["elapsed_pass"]
    assert decision["next_leaf"].startswith("U6.P8BH")
