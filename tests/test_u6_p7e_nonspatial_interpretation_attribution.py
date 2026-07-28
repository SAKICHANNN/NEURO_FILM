from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_nonspatial_attribution import (
    _neutral_axis,
    _render_arms,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p7e_nonspatial_interpretation_attribution_v1.json"
)


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p7e_contract_loads_five_fixed_nonspatial_arms() -> None:
    runtime = validate_contract(ROOT, _config())
    neutral = _neutral_axis(runtime, 33)
    assert list(neutral) == [
        "normalized_density",
        "channelwise_paper",
        "dye_matrix_only",
        "print_matrix_only",
        "full_print_control",
    ]
    assert all(row["minimum_lstar_step"] >= -1e-9 for row in neutral.values())


def test_p7e_arms_are_repeat_exact_bounded_and_distinct() -> None:
    runtime = validate_contract(ROOT, _config())
    source = np.random.default_rng(20260729).random((37, 41, 3))
    first = _render_arms(source, runtime)
    second = _render_arms(source, runtime)
    assert list(first) == [
        "colour_only",
        "normalized_density",
        "channelwise_paper",
        "dye_matrix_only",
        "print_matrix_only",
        "full_print_control",
    ]
    for arm_id in first:
        np.testing.assert_array_equal(first[arm_id], second[arm_id])
        assert np.all((first[arm_id] >= 0.0) & (first[arm_id] <= 1.0))
    assert len(
        {
            np.ascontiguousarray(values).tobytes()
            for values in first.values()
        }
    ) == len(first)
