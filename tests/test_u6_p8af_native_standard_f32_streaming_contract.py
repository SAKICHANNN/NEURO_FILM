from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.benchmark_u6_p8af_native_standard_f32_streaming import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u6_p8af_native_standard_f32_streaming_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8af_contract_binds_full_frame_f32_reference() -> None:
    config = json.loads(CONFIG.read_text())
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    parent = validate_contract(config)
    assert parent["result"]["output_sha256"] == config["reference"][
        "full_frame_output_sha256"
    ]
    assert parent["result"][
        "median_peak_process_tree_rss_bytes"
    ] == config["reference"]["median_peak_process_tree_rss_bytes"]
    assert config["execution"]["input_rows_generated_on_demand"]
    assert config["execution"]["output_rows_consumed_by_hash_sink"]
