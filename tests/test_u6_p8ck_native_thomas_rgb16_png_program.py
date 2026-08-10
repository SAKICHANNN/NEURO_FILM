from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_u6_p8ck_native_thomas_rgb16_png_program import (
    ROOT,
    _conformance,
    _json,
    _validate_contract,
)
from src.eval.native_msvc import sha256_file

CONTRACT = ROOT / "configs/u6_p8ck_native_thomas_rgb16_png_program_v1.json"
LLVM = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64"


def test_p8ck_contract_is_frozen() -> None:
    assert sha256_file(CONTRACT) == (
        "44b8ccde4cbb9d136df381bb32ac874c17e8351166bd6623ce5536d185ea3069"
    )
    _validate_contract(_json(CONTRACT))


@pytest.mark.skipif(not LLVM.is_dir(), reason="pinned LLVM-MinGW unavailable")
def test_p8ck_dual_compiler_png_conformance(tmp_path: Path) -> None:
    report = _conformance(_json(CONTRACT), tmp_path, LLVM / "bin/clang.exe")
    assert report["automatic_pass"] is True
    assert report["compiler_png_byte_exact"] is True
    assert all(row["repeat_exact"] for row in report["rows"].values())
    assert all(row["sample_exact"] for row in report["rows"].values())
    assert all(row["failure_exact"] for row in report["rows"].values())
