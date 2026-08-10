from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cj_native_thomas_rgb16_program_v1.json"


def test_p8cj_freezes_complete_freestanding_rgb16_program() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert contract["candidate"]["language"] == "freestanding-c11"
    assert contract["candidate"]["output_layout"] == "row-major-interleaved-rgb16"
    assert contract["candidate"]["row_partitions"] == [1, 7, 31, 128]
    assert contract["candidate"]["one_final_quantization"] is True
    assert contract["candidate"]["full_output_allowed"] is False


def test_p8cj_keeps_algorithm_and_claim_boundaries_closed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["candidate"]["amplitude_field_gauge_quantizer_sources_unchanged"] is True
    assert contract["candidate"]["model_profile_or_sample_change_allowed"] is False
    assert contract["conformance"]["shape_chw"] == [3, 193, 257]
    assert "not PNG encoding" in contract["claim_ceiling"]
