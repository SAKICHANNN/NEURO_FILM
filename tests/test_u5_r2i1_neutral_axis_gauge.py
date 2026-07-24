from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.neutral_axis_gauge import build_gauge, evaluate_gauge
from src.roll2film.sensitometry_gauge import NeutralAxisGaugeOperator


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict]:
    return (_load("u5_r2i1_neutral_axis_gauge_v1.json"), _load("u2_2b_sensitometry_print_composition_v1.json"), _load("u2_2a_sensitometry_primitive_v1.json"), _load("u5_r2e0_density_domain_operator_v1.json"))


def test_gauge_neutral_knots_replay_and_partition_are_exact() -> None:
    inputs = _inputs()
    operator = build_gauge(*inputs)
    neutral = np.linspace(0.0, 1.0, inputs[0]["gauge_knots"])
    values = np.stack((neutral, neutral, neutral), axis=-1)
    np.testing.assert_allclose(operator.apply(values), values, atol=2e-12, rtol=0.0)
    probes = np.random.default_rng(4).random((1000, 3))
    output = operator.apply(probes)
    replay = NeutralAxisGaugeOperator.from_dict(json.loads(json.dumps(operator.to_dict())))
    assert np.array_equal(replay.apply(probes), output)
    assert np.array_equal(np.concatenate([operator.apply(item) for item in np.array_split(probes, 7)]), output)


def test_gauge_guards_and_schema_fail_closed() -> None:
    operator = build_gauge(*_inputs())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        operator.apply(np.asarray([[1.01, 0.5, 0.5]]))
    payload = operator.to_dict()
    payload["schema"] = "tampered"
    with pytest.raises(ValueError, match="unsupported"):
        NeutralAxisGaugeOperator.from_dict(payload)


def test_frozen_gauge_audit_is_repeat_identical_and_passes() -> None:
    inputs = _inputs()
    first, first_arrays = evaluate_gauge(*inputs)
    second, second_arrays = evaluate_gauge(*inputs)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert all(np.array_equal(first_arrays[key], second_arrays[key]) for key in first_arrays)
    assert first["decision"] == "neutral_axis_gauge_pass"
    assert all(first["checks"].values())
