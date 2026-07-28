from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8o_row_linear import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8o_row_linear_resources_v1.json"


def test_p8o_contract_targets_only_attributed_roundtrip_peak() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    measurement = validate_contract(config)

    assert measurement["node"] == "U6.P8H"
    assert (
        config["implementation"]["roundtrip_linear_output"]
        == "one-preallocated-full-frame-float64"
    )
    assert (
        config["implementation"]["roundtrip_linear_execution"]
        == "row-bounded-eotf"
    )
    assert config["implementation"]["roundtrip_linear_tile_rows"] == 128
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]
