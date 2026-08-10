from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.native_granularity_amplitude_conformance import (
    NativeGranularityAmplitudeConformanceError,
    evaluate_conformance,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json"
CLANG = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def _contract() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_binds_p4bw_profile_without_fit() -> None:
    validate_contract(ROOT, _contract())
    assert _contract()["candidate"]["measurement_energy"] == 0.42674186556428034  # type: ignore[index]


def test_contract_rejects_spatial_scope_inflation() -> None:
    contract = _contract()
    contract["candidate"]["spatial_field_generation_allowed"] = True  # type: ignore[index]
    with pytest.raises(NativeGranularityAmplitudeConformanceError, match="contract drift"):
        validate_contract(ROOT, contract)


@pytest.mark.skipif(not CLANG.is_file(), reason="pinned LLVM-MinGW unavailable")
def test_dual_compiler_amplitude_conformance(tmp_path: Path) -> None:
    report = evaluate_conformance(ROOT, _contract(), output_dir=tmp_path, clang=CLANG)
    assert report["automatic_pass"] is True
    assert all(report["gate_results"].values())
    assert report["results"]["msvc"]["failure_atomic"] is True
    assert report["maximum_cross_compiler_absolute_error"] == 0.0
