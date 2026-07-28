from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8g_row_staged_context import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p8g_row_staged_context_resources_v1.json"
)


def test_p8g_contract_reuses_parent_measurement_without_pixel_change() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = validate_contract(config)
    assert parent["node"] == "U6.P8F"
    assert (
        config["implementation"]["source_context_density_execution"]
        == "row-staged"
    )
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]
