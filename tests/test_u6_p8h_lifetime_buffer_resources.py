from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8h_lifetime_buffers import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8h_lifetime_buffer_resources_v1.json"


def test_p8h_contract_requires_copy_free_hash_and_output_reuse() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = validate_contract(config)
    assert parent["node"] == "U6.P8G"
    assert not config["implementation"]["array_hash_copy_allowed"]
    assert config["implementation"][
        "display_output_reuses_consumed_float64_input"
    ]
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]
