from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8k_scene_context import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8k_scene_context_resources_v1.json"


def test_p8k_contract_requires_exact_scene_context_topology() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    measurement = validate_contract(config)

    assert measurement["node"] == "U6.P8H"
    assert (
        config["implementation"]["context_input"]
        == "scene-linear-row-staged"
    )
    assert not config["implementation"]["retained_full_encoded_source"]
    assert (
        config["implementation"]["physical_input"]
        == "one-full-frame-encoded-roundtrip-linear"
    )
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]
