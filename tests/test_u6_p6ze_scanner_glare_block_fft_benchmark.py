from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_u6_p6ze_scanner_glare_block_fft import (
    SCHEMA,
    numerical_probe,
    worker,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6ze_scanner_glare_block_fft_v1.json"


def test_contract_freezes_12mp_interleaved_benchmark() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == SCHEMA
    assert contract["benchmark"]["shape"] == [3000, 4000, 3]
    assert contract["benchmark"]["execution_order"] == [
        "fft-2d-reference",
        "block-fft-streaming",
        "block-fft-streaming",
        "fft-2d-reference",
    ]


@pytest.mark.parametrize("algorithm", ["fft-2d-reference", "block-fft-streaming"])
def test_small_worker_is_repeat_exact(algorithm: str) -> None:
    first = worker(algorithm, (31, 37, 3), seed=6206105, row_chunk=17)
    second = worker(algorithm, (31, 37, 3), seed=6206105, row_chunk=17)
    assert first["source_sha256"] == second["source_sha256"]
    assert first["output_sha256"] == second["output_sha256"]
    assert first["finite_bounded"] is True


def test_numerical_probe_passes_frozen_partitions() -> None:
    probe = numerical_probe(6206105, [257, 512, 769])
    assert probe["maximum_reference_absolute_error"] <= 1e-12
    assert probe["maximum_reference_rmse"] <= 1e-13
    assert probe["maximum_partition_absolute_error"] <= 1e-12


def test_worker_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        worker("unknown", (7, 9, 3), seed=6206105, row_chunk=3)
