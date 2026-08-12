from pathlib import Path

from src.eval.native_sensitometry_f64 import evaluate

ROOT = Path(__file__).resolve().parents[1]
LLVM = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def test_p4ex_native_sensitometry_f64(tmp_path: Path) -> None:
    result = evaluate(ROOT, ROOT / "configs/u6_p4ex_native_sensitometry_f64_v2.json", tmp_path, LLVM)
    assert result["automatic_pass"] is True
