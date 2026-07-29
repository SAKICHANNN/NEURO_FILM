from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark_u6_p3i_fft_backing_return import (
    SCHEMA,
    _field,
    worker,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p3i_fft_backing_return_benchmark_v1.json"


def test_contract_freezes_interleaved_two_by_three_benchmark() -> None:
    contract = json.loads(CONTRACT.read_text("utf-8"))
    assert contract["schema"] == SCHEMA
    assert contract["execution_order_per_shape"] == [
        "direct-separable",
        "fft-2d",
        "fft-2d",
        "direct-separable",
    ]
    assert contract["shapes"][-1] == [4000, 6000, 3]
    assert contract["production_integration_allowed"] is False


def test_field_is_repeat_exact_and_bounded() -> None:
    first = _field((31, 37, 3))
    second = _field((31, 37, 3))
    assert np.array_equal(first, second)
    assert first.dtype == np.float32
    assert float(np.min(first)) >= 0.0


@pytest.mark.parametrize("algorithm", ["direct-separable", "fft-2d"])
def test_small_worker_is_deterministic_and_direct_retaining(algorithm: str) -> None:
    first = worker(algorithm, (31, 37, 3))
    second = worker(algorithm, (31, 37, 3))
    assert first["source_sha256"] == second["source_sha256"]
    assert first["output_sha256"] == second["output_sha256"]
    assert first["finite_nonnegative"] is True
    assert first["minimum_direct_increment"] >= -2e-7


def test_worker_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        worker("unknown", (7, 9, 3))
