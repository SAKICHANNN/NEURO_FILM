from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_u6_p8ci_native_thomas_stream_program import evaluate

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8ci_native_thomas_stream_program_v1.json"
LLVM = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


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


def test_p8ci_executes_exact_dual_compiler_stream_program(tmp_path: Path) -> None:
    if not LLVM.is_file():
        pytest.skip("pinned LLVM-MinGW is unavailable")
    report = evaluate(CONTRACT, tmp_path, LLVM)
    assert report["automatic_pass"] is True
    assert report["decision"] == "retain_freestanding_native_thomas_stream_program"
    assert all(report["gate_results"].values())
    assert report["output_sha256"] == (
        "3bba3236551d89bc0eb37f4434740df5"
        "100f34dd7386cba668d82dce43eafe16"
    )
