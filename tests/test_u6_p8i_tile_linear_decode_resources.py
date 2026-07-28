from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8i_tile_linear_decode import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8i_tile_linear_decode_resources_v1.json"


def test_p8i_contract_requires_tile_decode_and_exact_pixels() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = validate_contract(config)

    assert parent["node"] == "U6.P8H"
    assert not config["implementation"]["full_frame_linear_decode_allowed"]
    assert (
        config["implementation"]["linear_decode_scope"]
        == "physical-halo-tile"
    )
    assert not config["implementation"][
        "per_element_decode_arithmetic_change_allowed"
    ]
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]
