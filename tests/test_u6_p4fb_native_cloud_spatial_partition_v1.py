from pathlib import Path

from src.eval.native_cloud_spatial_partition import evaluate

ROOT=Path(__file__).resolve().parents[1]
LLVM=ROOT/"outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def test_p4fb_partition(tmp_path:Path)->None:
    assert evaluate(ROOT,ROOT/"configs/u6_p4fb_native_cloud_spatial_partition_v1.json",tmp_path,LLVM)["automatic_pass"]
