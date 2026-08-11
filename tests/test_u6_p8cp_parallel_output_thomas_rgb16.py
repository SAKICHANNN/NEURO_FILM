from __future__ import annotations

import hashlib
from pathlib import Path

from src.eval.native_thomas_rgb16_cached_android_scale import (
    build_msvc_probe,
    run_host_probe,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cp_parallel_output_thomas_rgb16_v1.json"


def test_p8cp_contract_and_entry_point_are_frozen() -> None:
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == (
        "c4febe5b6bc897fa4970cd28054f477b00a8db992f04806fab11e941fb225e12"
    )
    header = ROOT / "native/film_physics/nf_thomas_rgb16_cached_f32_v1.h"
    assert "nf_thomas_rgb16_cached_f32_apply_parallel_output_v1" in header.read_text(
        encoding="utf-8"
    )


def test_parallel_output_is_exact_on_small_fixture(tmp_path: Path) -> None:
    host = build_msvc_probe(ROOT, tmp_path / "host")
    executable = Path(host["executable"])
    runs = [
        run_host_probe(
            executable,
            tmp_path / f"{tag}.raw",
            mode=mode,
            height=193,
            width=257,
            row_partition=128,
            parallel_layers=3,
        )
        for tag, mode in (
            ("cached", "cached"),
            ("output1", "cached_output3"),
            ("output2", "cached_output3"),
        )
    ]
    assert {row["raw_sha256"] for row in runs} == {
        "58e2a4187e3c4efdcb37620943878fd3e7ce0957eaa8e64f5932d47c4112a171"
    }
    assert runs[1]["facts"] == runs[2]["facts"]
    assert runs[1]["facts"]["invalid_calls"] == 0
