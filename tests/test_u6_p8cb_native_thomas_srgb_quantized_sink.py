from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_u6_p8cb_native_thomas_srgb_quantized_sink import (
    _validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cb_native_thomas_srgb_quantized_sink_v1.json"


def test_p8cb_contract_binds_exact_quantizer_and_parent() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    assert contract["quantizer"]["bit_depths"] == [8, 16]
    assert contract["quantizer"]["threshold_identity"] == (
        "fae645ef1aad04fcd1233631a32f820cf7696e3ca65d31939acf60d7f123674c"
    )


def test_p8cb_permits_exactly_one_final_quantization() -> None:
    candidate = json.loads(CONTRACT.read_text(encoding="utf-8"))["candidate"]
    assert candidate["one_final_quantization_only"] is True
    assert candidate["post_quantization_float_processing_allowed"] is False
    assert candidate["image_encoding_allowed"] is False
