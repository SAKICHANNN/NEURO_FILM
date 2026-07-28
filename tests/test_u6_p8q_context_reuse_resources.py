from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8q_context_reuse import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8q_context_reuse_resources_v1.json"


def test_p8q_contract_targets_joint_attributed_peaks() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    measurement = validate_contract(config)

    assert measurement["node"] == "U6.P8H"
    assert (
        config["implementation"]["source_context_schedule"]
        == "before-physical-execution"
    )
    assert (
        config["implementation"]["encoded_source_allocation"]
        == "destructively-reused-for-roundtrip-linear"
    )
    assert not config["implementation"][
        "pointwise_arithmetic_change_allowed"
    ]
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]
