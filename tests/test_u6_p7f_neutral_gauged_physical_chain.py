from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_neutral_gauged_chain import (
    _neutral_axis,
    _render_arms,
    apply_gauge_to_intermediate,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p7f_neutral_gauged_physical_chain_v1.json"


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p7f_reuses_exact_gauge_and_neutralizes_full_print() -> None:
    runtime, gauge = validate_contract(ROOT, _config())
    neutral = _neutral_axis(runtime, gauge, 16385)
    assert neutral["maximum_lab_chroma"] <= 1.0
    assert neutral["maximum_encoded_channel_spread"] <= 0.0001
    assert neutral["minimum_lstar_step"] >= -1e-9


def test_p7f_intermediate_gauge_matches_operator_nonspatial() -> None:
    runtime, gauge = validate_contract(ROOT, _config())
    source = np.random.default_rng(20260729).random((37, 41, 3))
    base = runtime.print_operator.apply(source)
    expected = gauge.apply(source)
    actual = apply_gauge_to_intermediate(base, gauge)
    np.testing.assert_array_equal(actual, expected)


def test_p7f_arms_are_repeat_exact_bounded_and_distinct() -> None:
    runtime, gauge = validate_contract(ROOT, _config())
    source = np.random.default_rng(20260730).random((37, 41, 3))
    first = _render_arms(source, runtime, gauge, sampling_dpi=4000)
    second = _render_arms(source, runtime, gauge, sampling_dpi=4000)
    assert list(first) == _config()["arms"]
    for arm_id in first:
        np.testing.assert_array_equal(first[arm_id], second[arm_id])
        assert np.all((first[arm_id] >= 0.0) & (first[arm_id] <= 1.0))
    assert len(
        {
            np.ascontiguousarray(values).tobytes()
            for values in first.values()
        }
    ) == len(first)
