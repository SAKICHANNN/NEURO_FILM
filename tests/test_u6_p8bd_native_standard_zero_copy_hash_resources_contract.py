from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8bd_native_standard_zero_copy_hash_resources import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/u6_p8bd_native_standard_zero_copy_hash_resources_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8bd_native_standard_zero_copy_hash_resources_decision_v1.json"
)


def test_p8bd_contract_changes_hashing_only() -> None:
    config = json.loads(CONTRACT.read_text())
    parent, base = validate_contract(config)
    assert parent["result"]["all_candidates_memory_fail"]
    assert not config["operator_changed"]
    assert not config["receipt_schema_changed"]
    assert not config["execution_policy_changed"]
    assert base["gates"]["maximum_peak_process_tree_rss_bytes"] == (
        384 * 1024 * 1024
    )


def test_p8bd_decision_passes_unchanged_resource_gates() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["result"]["status"] == "pass"
    assert decision["result"]["output_and_receipt_exact"]
    assert not decision["result"]["operator_changed"]
    assert not decision["result"]["receipt_schema_changed"]
    assert decision["result"]["memory_pass"]
    assert decision["result"]["elapsed_pass"]
    assert decision["next_leaf"].startswith("U6.P8BE")
