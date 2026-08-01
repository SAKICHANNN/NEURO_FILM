from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fp16_print_lut_compiler import _load_bound
from src.eval.streamed_explicit_print_runtime import (
    _operator,
    load_contract,
    run_stream,
)
from src.film_physics.print_runtime import apply_density_to_print_float32

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6w_streamed_explicit_print_runtime_v1.json"


def _print_operator():
    contract = load_contract(CONTRACT)
    p6v = _load_bound(ROOT, contract["parents"]["p6v_contract"])
    return _operator(ROOT, p6v)


def test_float32_runtime_tracks_float64_reference() -> None:
    operator = _print_operator()
    density = np.asarray([[0.89, 0.89, 0.89], [1.04, 1.04, 1.04]])
    candidate = apply_density_to_print_float32(operator, density)
    reference = operator.apply(density)
    assert candidate.dtype == np.float32
    assert np.max(np.abs(candidate.astype(np.float64) - reference)) <= 2e-6


def test_float32_runtime_rejects_outside_density() -> None:
    with pytest.raises(ValueError, match="outside"):
        apply_density_to_print_float32(_print_operator(), np.zeros((1, 3)))


def test_small_stream_is_partition_exact() -> None:
    operator = _print_operator()
    rows = [
        run_stream(
            operator,
            width=129,
            height=131,
            tile_rows=tile_rows,
            measure_performance=False,
        )
        for tile_rows in (17, 61, 131)
    ]
    assert len({row["output_sha256"] for row in rows}) == 1
    assert len({row["maximum_float32_vs_float64_absolute_error"] for row in rows}) == 1
    assert all(row["out_of_range_output_count"] == 0 for row in rows)
