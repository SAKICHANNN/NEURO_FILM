from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_reference_chain_android import build


LOCAL_NDK = Path(
    r"C:\Users\hhvrf\Documents\追色\outputs\tmp\tools\android-ndk-r27d"
)


def test_pinned_android_arm64_and_x86_64_link(tmp_path: Path) -> None:
    if not LOCAL_NDK.is_dir():
        pytest.skip("pinned local Android NDK is unavailable")
    report = build(LOCAL_NDK, tmp_path)
    assert report["claim_scope"] == "cross-compile and link; no device execution"
    assert report["ndk"]["revision"] == "27.3.13750724"
    assert set(report["artifacts"]) == {"arm64-v8a", "x86_64"}
    assert report["artifacts"]["arm64-v8a"]["elf_machine"] == "AArch64"
    assert (
        report["artifacts"]["x86_64"]["elf_machine"]
        == "Advanced Micro Devices X86-64"
    )
    assert all(
        artifact["bytes"] > 0 and len(artifact["sha256"]) == 64
        for artifact in report["artifacts"].values()
    )
