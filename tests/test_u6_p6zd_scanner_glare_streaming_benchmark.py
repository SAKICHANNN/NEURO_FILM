from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark_u6_p6zd_scanner_glare_streaming import (
    SCHEMA,
    _field,
    _numerical_probe,
    worker,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6zd_scanner_glare_streaming_benchmark_v1.json"


def test_contract_freezes_interleaved_reference_streaming_benchmark() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == SCHEMA
    assert contract["execution_order_per_shape"] == [
        "fft-2d-reference",
        "separable-row-streaming",
        "separable-row-streaming",
        "fft-2d-reference",
    ]
    assert contract["shapes"][-1] == [3000, 4000, 3]
    assert contract["row_chunk"] == 64
    assert contract["production_integration_allowed"] is False


def test_field_is_repeat_exact_float64_and_bounded() -> None:
    first = _field((31, 37, 3), 6206104)
    second = _field((31, 37, 3), 6206104)
    assert np.array_equal(first, second)
    assert first.dtype == np.float64
    assert float(np.min(first)) >= 0.0
    assert float(np.max(first)) <= 1.0


@pytest.mark.parametrize("algorithm", ["fft-2d-reference", "separable-row-streaming"])
def test_small_worker_is_repeat_exact_and_bounded(algorithm: str) -> None:
    first = worker(algorithm, (31, 37, 3), seed=6206104, row_chunk=17)
    second = worker(algorithm, (31, 37, 3), seed=6206104, row_chunk=17)
    assert first["source_sha256"] == second["source_sha256"]
    assert first["output_sha256"] == second["output_sha256"]
    assert first["finite_bounded"] is True


def test_numerical_probe_matches_frozen_error_budget() -> None:
    probe = _numerical_probe(6206104, 64)
    assert probe["maximum_absolute_error"] <= 1e-12
    assert probe["rmse"] <= 1e-13


def test_worker_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        worker("unknown", (7, 9, 3), seed=6206104, row_chunk=3)
