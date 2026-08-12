from pathlib import Path

from src.eval.native_cloud_row_chain_24mp import evaluate

ROOT=Path(__file__).resolve().parents[1]
LLVM=ROOT/"outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def test_p4es_native_24mp(tmp_path: Path) -> None:
    result=evaluate(ROOT,ROOT/"configs/u6_p4es_native_cloud_row_chain_24mp_v1.json",tmp_path,LLVM)
    assert result["stable"]["gates"]["repeat"] is True
