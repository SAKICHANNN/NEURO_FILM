from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8ci_native_thomas_stream_program_v1.json"


def test_p8ci_freezes_exact_freestanding_stream_program() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert contract["candidate"]["language"] == "freestanding-c11"
    assert contract["candidate"]["row_partitions"] == [1, 7, 31, 128]
    assert contract["candidate"]["field_density_and_reducer_sources_unchanged"] is True
    assert contract["candidate"]["model_profile_or_sample_change_allowed"] is False
    assert contract["conformance"]["shape_chw"] == [3, 193, 257]


def test_p8ci_keeps_streaming_failure_and_claim_boundaries_explicit() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["candidate"]["callback_may_retain_rows"] is False
    assert contract["candidate"]["failure_stops_before_next_callback"] is True
    assert contract["gates"]["require_callback_failure_propagation"] is True
    assert "not RGB orchestration" in contract["claim_ceiling"]
