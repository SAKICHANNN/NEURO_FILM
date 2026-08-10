from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_u6_p8ca_native_thomas_gauged_sink import (
    _gauge_payload,
    _validate_contract,
)
from src.film_physics.native_gauge_profile import native_gauge_payload_sha256

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8ca_native_thomas_gauged_sink_v1.json"


def test_p8ca_contract_binds_exact_parents_and_gauge() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    assert native_gauge_payload_sha256(_gauge_payload()) == contract["candidate"][
        "neutral_gauge_payload_sha256"
    ]
    assert contract["candidate"]["domain_order"][-1] == (
        "neutral-axis-display-linear-gauge"
    )


def test_p8ca_forbids_premature_output_work() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["candidate"]["full_output_or_channel_plane_allowed"] is False
    assert (
        contract["candidate"]["output_transfer_quantization_or_encoding_allowed"]
        is False
    )
