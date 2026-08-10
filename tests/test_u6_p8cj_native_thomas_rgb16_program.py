from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_u6_p8cj_native_thomas_rgb16_program import evaluate

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cj_native_thomas_rgb16_program_v1.json"
LLVM = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


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


def test_p8cj_executes_exact_dual_compiler_rgb16_program(tmp_path: Path) -> None:
    if not LLVM.is_file():
        pytest.skip("pinned LLVM-MinGW is unavailable")
    report = evaluate(CONTRACT, tmp_path, LLVM)
    assert report["automatic_pass"] is True
    assert report["decision"] == "retain_freestanding_native_thomas_rgb16_program"
    assert all(report["gate_results"].values())
    assert report["output_sha256"] == (
        "20d84ec2f239c4f87732476030f700a3f"
        "43f22fec4e59504bbe16654305d39cb"
    )
