from pathlib import Path

from src.eval.native_cloud_scale_window_24mp import evaluate

ROOT=Path(__file__).resolve().parents[1]
LLVM=ROOT/"outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def test_p4eu_developed_window_24mp(tmp_path: Path) -> None:
    result=evaluate(ROOT,ROOT/"configs/u6_p4eu_native_cloud_developed_window_v4.json",tmp_path,LLVM,developed=True)
    assert result["automatic_pass"] is False
    assert result["stable"]["gates"]["scale"] is True
    assert result["stable"]["gates"]["output"] is False
