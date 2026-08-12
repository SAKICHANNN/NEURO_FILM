from pathlib import Path

import pytest

from scripts.build_u6_p4ea_native_cloud_attenuation_portable_v1 import build

ROOT = Path(__file__).resolve().parents[1]
NDK = Path(r"D:\nf-019f4b76-android\android-ndk-r27d")
LLVM = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64"


@pytest.mark.skipif(
    not NDK.is_dir() or not LLVM.is_dir(), reason="portable toolchains unavailable"
)
def test_p4ea_repeat_portable_build(tmp_path: Path) -> None:
    contract = ROOT / "configs/u6_p4ea_native_cloud_attenuation_portable_build_v1.json"
    first = build(contract, NDK, LLVM, tmp_path / "a")
    second = build(contract, NDK, LLVM, tmp_path / "b")
    assert first == second
