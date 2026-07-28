from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8u_packed_lab import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8u_packed_lab_resources_v1.json"


def test_p8u_contract_targets_only_attributed_lab_storage() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    measurement = validate_contract(config)

    assert measurement["node"] == "U6.P8H"
    assert (
        config["implementation"]["source_context_lab_storage"]
        == "first-half-of-consumed-float64-encoded-allocation"
    )
    assert (
        config["implementation"]["source_context_reduction"]
        == "exact-legacy-numpy-mean-std"
    )
    assert (
        config["implementation"]["roundtrip_linear_refill"]
        == "row-local-exact-oetf-then-eotf"
    )
    assert not config["implementation"][
        "pointwise_arithmetic_change_allowed"
    ]
    assert not config["implementation"][
        "final_pixel_identity_change_allowed"
    ]
