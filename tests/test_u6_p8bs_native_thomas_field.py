from __future__ import annotations

import ctypes
import json
from pathlib import Path

import pytest

from src.eval.native_thomas_field_conformance import (
    NativeThomasConformanceError,
    evaluate_conformance,
    profile_from_contract,
    validate_contract,
)
from src.film_physics.native_thomas_field import NativeThomasFieldProfileV1

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8bs_native_thomas_field_v1.json"
CLANG = (
    ROOT
    / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"
)


def _contract() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_binds_retained_parents() -> None:
    validate_contract(ROOT, _contract())
    profile = profile_from_contract(_contract())
    assert profile.component_seeds == (
        2611923443488327891,
        11400714819323198485,
    )
    assert ctypes.sizeof(NativeThomasFieldProfileV1) == 64


def test_contract_rejects_capacity_drift() -> None:
    contract = _contract()
    contract["candidate"]["additional_model_capacity_allowed"] = True  # type: ignore[index]
    with pytest.raises(NativeThomasConformanceError, match="contract drift"):
        validate_contract(ROOT, contract)


@pytest.mark.skipif(not CLANG.is_file(), reason="pinned LLVM-MinGW unavailable")
def test_dual_compiler_conformance(tmp_path: Path) -> None:
    report = evaluate_conformance(ROOT, _contract(), output_dir=tmp_path, clang=CLANG)
    assert report["automatic_pass"] is True
    assert all(report["gate_results"].values())
    assert report["results"]["msvc"]["repeat_exact"] is True
    assert report["results"]["llvm_mingw"]["failure_atomic"] is True
    assert report["maximum_cross_compiler_absolute_error"] == 0.0
