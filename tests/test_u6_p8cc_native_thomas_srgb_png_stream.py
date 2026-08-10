from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cc_native_thomas_srgb_png_stream_v1.json"


def test_p8cc_contract_freezes_streaming_png_semantics() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert contract["encoder"]["bit_depths"] == [8, 16]
    assert contract["encoder"]["icc_profile_bytes"] == 588
    assert contract["candidate"]["full_output_allowed"] is False
    assert contract["candidate"]["one_final_quantization_only"] is True


def test_p8cc_contract_keeps_claim_below_device_and_product() -> None:
    ceiling = json.loads(CONTRACT.read_text(encoding="utf-8"))["claim_ceiling"]
    assert "not device runtime" in ceiling
    assert "not" in ceiling and "product authorization" in ceiling
