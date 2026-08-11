from __future__ import annotations

import hashlib
from pathlib import Path

from src.eval.native_thomas_rgb16_png_android_scale import build_msvc_probe
from src.eval.native_thomas_rgb16_png_parallel_android_scale import (
    parse_facts,
    run_host_probe,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cq_parallel_thomas_rgb16_png_v1.json"


def test_p8cq_contract_and_entry_points_are_frozen() -> None:
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == (
        "2132d0d4d1f69cec937ab11091a84cd4773ad7566e4b4fa51fe4b982d18380bc"
    )
    header = ROOT / "native/film_physics/nf_thomas_rgb16_png_f32_v1.h"
    text = header.read_text(encoding="utf-8")
    assert "nf_thomas_rgb16_png_cached_parallel_workspace_bytes_v1" in text
    assert "nf_thomas_rgb16_png_cached_parallel_apply_v1" in text


def test_parallel_png_facts_are_strict() -> None:
    facts = parse_facts(
        "mode=parallel status=0 height=193 width=257 rows=128 bytes=123 calls=4 "
        "workspace=456 means=0x1p-12,0x1p-12,0x1p-12 invalid_status=3 "
        "invalid_bytes=0 invalid_calls=0 invalid_means=-0x1.ap+3,-0x1.ap+3,-0x1.ap+3"
    )
    assert facts["status"] == 0
    assert facts["invalid_status"] == 3
    assert facts["invalid_bytes"] == 0
    assert facts["invalid_calls"] == 0


def test_parallel_png_small_host_repeats_exactly(tmp_path: Path) -> None:
    build = build_msvc_probe(ROOT, tmp_path / "host")
    executable = Path(build["executable"])
    first = run_host_probe(
        executable,
        tmp_path / "first.png",
        height=193,
        width=257,
        row_partition=128,
    )
    second = run_host_probe(
        executable,
        tmp_path / "second.png",
        height=193,
        width=257,
        row_partition=128,
    )
    stable_keys = (
        "facts",
        "png_bytes",
        "png_sha256",
        "decoded_sha256",
        "icc_sha256",
        "icc_exact",
    )
    assert {key: first[key] for key in stable_keys} == {
        key: second[key] for key in stable_keys
    }
    assert first["facts"]["invalid_calls"] == 0
