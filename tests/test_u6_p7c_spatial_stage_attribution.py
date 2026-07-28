from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_joint_ablation import load_contracts
from src.eval.physical_spatial_stage_attribution import (
    ARMS,
    render_cumulative_stages,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p7c_spatial_stage_attribution_v1.json"


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p7c_contract_binds_closed_p7b() -> None:
    runtime_parent = validate_contract(ROOT, _config())
    assert runtime_parent["node"] == "U6.P7A3"


def test_cumulative_stages_are_repeat_exact_and_bounded() -> None:
    runtime_parent = validate_contract(ROOT, _config())
    _, runtime = load_contracts(ROOT, runtime_parent)
    source = np.random.default_rng(20260728).random((37, 41, 3))
    first_pre, first_post = render_cumulative_stages(source, runtime)
    second_pre, second_post = render_cumulative_stages(source, runtime)
    assert tuple(first_pre) == ARMS
    assert tuple(first_post) == ARMS
    for name in ARMS:
        np.testing.assert_array_equal(first_pre[name], second_pre[name])
        np.testing.assert_array_equal(first_post[name], second_post[name])
        assert np.all((first_pre[name] >= 0.0) & (first_pre[name] <= 1.0))
        assert np.all((first_post[name] >= 0.0) & (first_post[name] <= 1.0))


def test_each_spatial_stage_changes_the_synthetic_frame() -> None:
    runtime_parent = validate_contract(ROOT, _config())
    _, runtime = load_contracts(ROOT, runtime_parent)
    source = np.full((65, 67, 3), 0.18, dtype=np.float64)
    source[32, 33] = (0.82, 0.24, 0.08)
    pre, _ = render_cumulative_stages(source, runtime)
    for previous, current in zip(ARMS[:-1], ARMS[1:], strict=True):
        assert not np.array_equal(pre[previous], pre[current])
