from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_u6_p8ck_native_thomas_rgb16_png_program import (
    ROOT,
    _conformance,
    _json,
    _stable_payload,
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


def test_p8ck_stable_copy_does_not_alias_runtime_evidence() -> None:
    report = {
        "conformance": {"toolchains": {"msvc": {"dll_path": "x"}}},
        "performance": {
            "wall_seconds": [1.0],
            "peak_process_tree_rss_bytes": [2],
            "wall_repeat_ratio": 1.0,
            "rss_repeat_ratio": 1.0,
            "runs": [
                {
                    "peak_process_tree_rss_bytes": 2,
                    "worker": {"wall_seconds": 1.0},
                }
            ],
        },
    }
    stable = _stable_payload(report)
    assert report["performance"]["wall_seconds"] == [1.0]
    assert report["performance"]["runs"][0]["worker"]["wall_seconds"] == 1.0
    assert "wall_seconds" not in stable["performance"]
    assert "dll_path" not in stable["conformance"]["toolchains"]["msvc"]


@pytest.mark.skipif(not LLVM.is_dir(), reason="pinned LLVM-MinGW unavailable")
def test_p8ck_dual_compiler_png_conformance(tmp_path: Path) -> None:
    report = _conformance(_json(CONTRACT), tmp_path, LLVM / "bin/clang.exe")
    assert report["automatic_pass"] is True
    assert report["compiler_png_byte_exact"] is True
    assert all(row["repeat_exact"] for row in report["rows"].values())
    assert all(row["sample_exact"] for row in report["rows"].values())
    assert all(row["failure_exact"] for row in report["rows"].values())
