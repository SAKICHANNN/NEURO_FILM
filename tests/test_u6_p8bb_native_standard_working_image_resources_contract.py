from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8bb_native_standard_working_image_resources import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/u6_p8bb_native_standard_working_image_resources_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8bb_native_standard_working_image_resources_decision_v1.json"
)


def test_p8bb_resource_contract_binds_runtime_and_frozen_gates() -> None:
    config = json.loads(CONTRACT.read_text())
    package = validate_contract(config)
    assert package["package_id"].endswith("win-x64")
    assert config["gates"]["maximum_peak_process_tree_rss_bytes"] == (
        384 * 1024 * 1024
    )
    assert config["gates"]["maximum_worker_elapsed_seconds"] == 15.0
    assert config["execution"]["full_frame_working_image_retained"]
    assert config["execution"]["output_staging_fsync"]
    assert not config["execution"]["decoder_or_encoder_included"]


def test_p8bb_decision_preserves_failed_memory_gate() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["result"]["status"] == (
        "closed-memory-gate-failed"
    )
    assert decision["result"]["output_repeat_and_p8ay_exact"]
    assert not decision["result"]["memory_pass"]
    assert decision["result"]["elapsed_pass"]
    assert decision["next_leaf"].startswith("U6.P8BC")
