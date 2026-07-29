from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p3k_streamed_fft_backing_return import (
    SCHEMA,
    worker,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs" / "u6_p3k_streamed_fft_backing_return_benchmark_v1.json"
)


def test_contract_freezes_24mp_tile512_without_full_output() -> None:
    contract = json.loads(CONTRACT.read_text("utf-8"))
    assert contract["schema"] == SCHEMA
    assert contract["shape"] == [4000, 6000, 3]
    assert contract["tile_rows"] == 512
    assert contract["automatic_gates"]["maximum_peak_process_tree_rss_bytes"] == 2**30
    assert contract["production_integration_allowed"] is False


def test_small_stream_worker_repeats_without_full_output() -> None:
    first = worker((31, 37, 3), 11)
    second = worker((31, 37, 3), 11)
    assert first["source_sha256"] == second["source_sha256"]
    assert first["output_sha256"] == second["output_sha256"]
    assert first["coverage_exactly_once"] is True
    assert first["maximum_live_output_core_rows"] == 11
    assert first["finite_nonnegative_and_direct_retaining"] is True
    assert first["full_output_allocated"] is False
    assert first["logical_output_bytes"] == 31 * 37 * 3 * 4
