from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.native_thomas_rgb16_png_android_runtime import (
    build_android_probe,
    build_host_probe,
    run_host_probe,
)

ROOT = Path(__file__).resolve().parents[1]
NDK = Path(r"D:\nf-019f4b76-android\android-ndk-r27d")
HOST_CLANG = (
    ROOT
    / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"
)


def test_probe_builds_and_host_replays_exactly(tmp_path: Path) -> None:
    if not NDK.is_dir() or not HOST_CLANG.is_file():
        pytest.skip("owned target toolchains are unavailable")
    host1 = tmp_path / "host1/probe.exe"
    host2 = tmp_path / "host2/probe.exe"
    android1 = tmp_path / "android1/probe"
    android2 = tmp_path / "android2/probe"
    assert build_host_probe(ROOT, HOST_CLANG, host1) == build_host_probe(
        ROOT, HOST_CLANG, host2
    )
    assert build_android_probe(ROOT, NDK, android1) == build_android_probe(
        ROOT, NDK, android2
    )
    first = run_host_probe(host1, tmp_path / "first.png")
    second = run_host_probe(host1, tmp_path / "second.png")
    assert first == second
    assert first["png_bytes"] == 16467
    assert first["png_sha256"] == (
        "7737cac78befc65cec140fd5c22c89a7b750faca24f18781896709a70f6ba31a"
    )
    assert "invalid_status=3 invalid_calls=0" in first["stdout"]
    assert "failed_status=4 failed_calls=2" in first["stdout"]
