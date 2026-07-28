from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8s_joint_buffer_reuse import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8s_joint_buffer_reuse_resources_v1.json"


def test_p8s_contract_targets_both_tied_full_frame_copies() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    measurement = validate_contract(config)

    assert measurement["node"] == "U6.P8H"
    assert (
        config["implementation"]["source_context_float32_cast"]
        == "row-local-only"
    )
    assert (
        config["implementation"]["physical_output"]
        == "destructively-reuses-consumed-linear-buffer"
    )
    assert config["implementation"][
        "forward_and_reverse_exact_required"
    ]
    assert not config["implementation"][
        "pointwise_arithmetic_change_allowed"
    ]
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]
