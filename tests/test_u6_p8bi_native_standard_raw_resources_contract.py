from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8bi_native_standard_raw_resources import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p8bi_native_standard_raw_resources_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8bi_native_standard_raw_resources_decision_v1.json"
)


def test_p8bi_contract_binds_exact_raw_and_resource_gates() -> None:
    config = json.loads(CONTRACT.read_text())
    parent, _ = validate_contract(config)
    assert parent["real_raw_smoke"]["sha256"] == (
        config["raw_fixture_sha256"]
    )
    assert config["repeats"] == 2
    assert config["gates"]["maximum_peak_process_tree_rss_bytes"] == (
        1024 * 1024 * 1024
    )
    assert not config["production_default_changed"]


def test_p8bi_decision_preserves_generic_raw_claim_ceiling() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["result"]["status"] == "pass"
    assert decision["result"]["raw_to_png_repeat_exact"]
    assert decision["result"]["memory_pass"]
    assert decision["result"]["elapsed_pass"]
    assert "generic rawpy" in decision["claim_ceiling"]
    assert decision["next_leaf"].startswith("U6.P8BJ")
