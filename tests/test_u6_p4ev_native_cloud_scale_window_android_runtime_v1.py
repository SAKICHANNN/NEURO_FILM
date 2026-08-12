from pathlib import Path

import pytest

from src.eval.native_cloud_scale_window_android_runtime import evaluate

ROOT = Path(__file__).resolve().parents[1]
NDK = Path(r"D:\nf-019f4b76-android\android-ndk-r27d")
LLVM = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64"
SDK = Path(r"D:\nf-019f4b76-android\runtime-sdk")
AVD = Path(r"D:\nf-019f4b76-android\avd")


@pytest.mark.skipif(
    not all(path.is_dir() for path in (NDK, LLVM, SDK, AVD)),
    reason="Android runtime unavailable",
)
def test_p4ev_android_runtime(tmp_path: Path) -> None:
    result = evaluate(
        ROOT,
        ROOT / "configs/u6_p4ev_native_cloud_scale_window_android_runtime_v1.json",
        NDK,
        LLVM / "bin/clang.exe",
        SDK,
        AVD,
        tmp_path,
    )
    assert result["automatic_pass"] is True
