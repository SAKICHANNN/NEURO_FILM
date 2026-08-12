from pathlib import Path

import pytest

from src.eval.deterministic_log10_conformance import evaluate

ROOT=Path(__file__).resolve().parents[1]
NDK=Path(r"D:\nf-019f4b76-android\android-ndk-r27d")
LLVM=ROOT/"outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64"
SDK=Path(r"D:\nf-019f4b76-android\runtime-sdk")
AVD=Path(r"D:\nf-019f4b76-android\avd")


@pytest.mark.skipif(not all(p.is_dir() for p in (NDK,LLVM,SDK,AVD)),reason="Android runtime unavailable")
def test_p4eq_deterministic_log10(tmp_path: Path) -> None:
    assert evaluate(ROOT,ROOT/"configs/u6_p4eq_deterministic_log10f_v1.json",tmp_path,LLVM/"bin/clang.exe",NDK,SDK,AVD)["automatic_pass"] is True
