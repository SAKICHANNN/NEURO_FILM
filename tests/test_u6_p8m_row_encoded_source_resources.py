from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8m_row_encoded_source import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8m_row_encoded_source_resources_v1.json"


def test_p8m_contract_targets_only_attributed_encoded_source_peak() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    measurement = validate_contract(config)

    assert measurement["node"] == "U6.P8H"
    assert (
        config["implementation"]["encoded_source_output"]
        == "one-preallocated-full-frame-float64"
    )
    assert (
        config["implementation"]["encoded_source_input_cast"]
        == "row-bounded-float64"
    )
    assert config["implementation"]["encoded_source_tile_rows"] == 128
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]
