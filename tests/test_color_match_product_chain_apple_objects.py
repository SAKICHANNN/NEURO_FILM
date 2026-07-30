from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_reference_chain_apple_objects import build


TOOLCHAIN = (
    ROOT.parent
    / "追色"
    / "outputs"
    / "tmp"
    / "tools"
    / "llvm-mingw-20260616-ucrt-x86_64"
)


def test_pinned_clang_emits_apple_arm64_core_objects(
    tmp_path: Path,
) -> None:
    if not TOOLCHAIN.is_dir():
        pytest.skip("pinned local LLVM toolchain is unavailable")
    report = build(TOOLCHAIN, tmp_path)
    assert report["claim_scope"] == (
        "freestanding Apple ARM64 object compilation only"
    )
    assert report["toolchain"]["llvm_version"] == "22.1.8"
    assert set(report["artifacts"]) == {"macos-arm64", "ios-arm64"}
    assert all(
        artifact["format"] == "Mach-O arm64"
        and artifact["file_type"] == "relocatable-object"
        and artifact["bytes"] > 0
        and len(artifact["sha256"]) == 64
        for artifact in report["artifacts"].values()
    )
    assert any(
        "no link, load" in limitation
        for limitation in report["limitations"]
    )
