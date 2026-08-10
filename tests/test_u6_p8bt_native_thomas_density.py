from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.native_thomas_density_conformance import (
    NativeThomasDensityConformanceError,
    evaluate_conformance,
    fixture_arrays,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8bt_native_density_thomas_transmittance_v1.json"
CLANG = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def _contract() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_binds_physical_parents() -> None:
    validate_contract(ROOT, _contract())
    base, sigma = fixture_arrays((193, 257))
    assert base.dtype.name == "float32"
    assert sigma.shape == base.shape
    assert float(base.min()) > 0.0
    assert float(sigma.min()) > 0.0


def test_contract_rejects_display_rgb_noise() -> None:
    contract = _contract()
    contract["candidate"]["display_rgb_noise_allowed"] = True  # type: ignore[index]
    with pytest.raises(NativeThomasDensityConformanceError, match="contract drift"):
        validate_contract(ROOT, contract)


@pytest.mark.skipif(not CLANG.is_file(), reason="pinned LLVM-MinGW unavailable")
def test_dual_compiler_density_conformance(tmp_path: Path) -> None:
    report = evaluate_conformance(ROOT, _contract(), output_dir=tmp_path, clang=CLANG)
    assert report["automatic_pass"] is True
    assert all(report["gate_results"].values())
    assert report["results"]["msvc"]["failure_atomic"] is True
    assert report["results"]["llvm_mingw"]["repeat_exact"] is True
