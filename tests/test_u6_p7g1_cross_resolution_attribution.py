from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_cross_resolution_attribution import (
    render_stages,
    validate_contract,
)
from src.eval.physical_neutral_gauged_invariance import _resize


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p7g1_cross_resolution_attribution_v1.json"
)


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p7g1_contract_binds_resolution_failure_only() -> None:
    runtime, _ = validate_contract(ROOT, _config())
    assert set(_config()["sample_ids"]) <= set(runtime.eligible_ids)


def test_stage_renderer_is_repeat_exact_and_shape_stable() -> None:
    runtime, gauge = validate_contract(ROOT, _config())
    source = np.random.default_rng(9).random((37, 53, 3))
    first = render_stages(
        source, runtime, gauge, sampling_dpi=4000
    )
    second = render_stages(
        source, runtime, gauge, sampling_dpi=4000
    )
    assert set(first) == set(_config()["stages"])
    assert all(
        value.shape == source.shape and np.array_equal(value, second[name])
        for name, value in first.items()
    )


def test_source_control_is_exact_by_construction() -> None:
    runtime, gauge = validate_contract(ROOT, _config())
    source = np.random.default_rng(11).random((48, 72, 3))
    high = render_stages(
        source, runtime, gauge, sampling_dpi=4000
    )
    low_source = _resize(source, (24, 36))
    low = render_stages(
        low_source, runtime, gauge, sampling_dpi=2000
    )
    assert np.array_equal(
        low["source_encoded_control"],
        _resize(high["source_encoded_control"], (24, 36)),
    )
