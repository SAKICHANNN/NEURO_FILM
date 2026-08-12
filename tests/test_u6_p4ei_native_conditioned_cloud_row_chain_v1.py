from pathlib import Path

import pytest

from src.eval.native_conditioned_cloud_row_chain import evaluate

ROOT = Path(__file__).resolve().parents[1]
CLANG = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


@pytest.mark.skipif(not CLANG.is_file(), reason="pinned LLVM unavailable")
def test_p4ei_native_conditioned_cloud_chain(tmp_path: Path) -> None:
    report = evaluate(ROOT, ROOT / "configs/u6_p4ei_native_conditioned_cloud_row_chain_v1.json", tmp_path, CLANG)
    assert report["automatic_pass"] is True
