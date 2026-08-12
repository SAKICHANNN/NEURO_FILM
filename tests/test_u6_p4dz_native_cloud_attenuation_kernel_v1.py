from pathlib import Path

import pytest

from src.eval.native_cloud_attenuation_conformance import evaluate

ROOT = Path(__file__).resolve().parents[1]
CLANG = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


@pytest.mark.skipif(not CLANG.is_file(), reason="pinned LLVM unavailable")
def test_p4dz_dual_compiler_execution(tmp_path: Path) -> None:
    report = evaluate(
        ROOT,
        ROOT / "configs/u6_p4dz_native_cloud_attenuation_kernel_v1.json",
        tmp_path,
        CLANG,
    )
    assert report["automatic_pass"] is True
