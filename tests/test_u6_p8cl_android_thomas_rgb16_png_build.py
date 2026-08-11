from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.native_thomas_rgb16_png_android_conformance import evaluate

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cl_android_thomas_rgb16_png_build_v1.json"
LOCAL_NDK = Path(r"D:\nf-019f4b76-android\android-ndk-r27d")


def test_android_dual_abi_build_is_reproducible(tmp_path: Path) -> None:
    if not LOCAL_NDK.is_dir():
        pytest.skip("owned pinned Android NDK r27d is unavailable")
    report = evaluate(CONTRACT, LOCAL_NDK, tmp_path)
    assert report["automatic_pass"] is True
    assert report["decision"] == "retain_android_dual_abi_build_for_p8ck"
    assert set(report["abis"]) == {"arm64-v8a", "x86_64"}
    assert all(row["byte_exact_rebuild"] for row in report["abis"].values())
    assert report["claim_ceiling"].startswith(
        "Android arm64-v8a and x86_64 compile/link"
    )
