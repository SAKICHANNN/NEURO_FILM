from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8j_context_first import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8j_context_first_resources_v1.json"


def test_p8j_contract_requires_exact_context_first_schedule() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    measurement = validate_contract(config)

    assert measurement["node"] == "U6.P8H"
    assert (
        config["implementation"]["source_context_schedule"]
        == "before-linear-and-gauged-full-frame-allocation"
    )
    assert not config["implementation"][
        "source_context_rebuilt_during_display"
    ]
    assert (
        config["implementation"]["physical_linear_decode_mode"]
        == "restored-full-frame-p8h"
    )
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]
