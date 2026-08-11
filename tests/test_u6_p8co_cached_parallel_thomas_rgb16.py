from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.eval.native_thomas_rgb16_cached_android_scale import (
    build_android_probe,
    build_msvc_probe,
    parse_facts,
    run_host_probe,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8co_cached_parallel_thomas_rgb16_v1.json"
NDK = Path(r"D:\nf-019f4b76-android\android-ndk-r27d")


def test_contract_and_cached_abi_are_frozen() -> None:
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == (
        "982d75776173844ff1d2c4b2ad0fe8e0d6c17a123e94a97d1eab393c68f252e5"
    )
    header = ROOT / "native/film_physics/nf_thomas_rgb16_cached_f32_v1.h"
    assert "NF_THOMAS_RGB16_CACHED_F32_ABI_VERSION_V1 1u" in header.read_text(
        encoding="utf-8"
    )


def test_cached_probe_facts_are_strict() -> None:
    facts = parse_facts(
        "mode=cached status=0 height=193 width=257 rows=128 parallel=3 "
        "values=148803 calls=2 workspace=3950604 means=0x1p-12,0x1p-12,0x1p-12 "
        "invalid_status=3 invalid_calls=0 invalid_means=-0x1.ap+3,-0x1.ap+3,-0x1.ap+3"
    )
    assert facts["mode"] == "cached"
    assert facts["parallel"] == 3
    assert facts["invalid_status"] == 3
    assert facts["invalid_calls"] == 0


def test_cached_probe_builds_and_small_outputs_are_exact(tmp_path: Path) -> None:
    if not NDK.is_dir():
        pytest.skip("owned Android NDK is unavailable")
    host = build_msvc_probe(ROOT, tmp_path / "host")
    android = build_android_probe(ROOT, NDK, tmp_path / "android/probe")
    assert Path(host["executable"]).is_file()
    assert Path(android["executable"]).is_file()
    executable = Path(host["executable"])
    runs = [
        run_host_probe(
            executable,
            tmp_path / f"{tag}.raw",
            mode=mode,
            height=193,
            width=257,
            row_partition=128,
            parallel_layers=parallel,
        )
        for tag, mode, parallel in (
            ("legacy", "legacy", 1),
            ("cached1", "cached", 1),
            ("cached3", "cached", 3),
        )
    ]
    assert {row["raw_sha256"] for row in runs} == {
        "58e2a4187e3c4efdcb37620943878fd3e7ce0957eaa8e64f5932d47c4112a171"
    }
    assert all(row["facts"]["invalid_calls"] == 0 for row in runs)
