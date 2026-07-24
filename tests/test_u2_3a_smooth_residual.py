from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.smooth_residual_composition import _cyclic_lut, evaluate_composition
from src.roll2film.constrained import LUTConstraintSpec, identity_lut
from src.roll2film.residual import SensitometryResidualLUTOperator
from src.eval.sensitometry_print_composition import build_composition


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict]:
    return (
        _load("u2_3a_smooth_residual_composition_v1.json"),
        _load("u2_2b_sensitometry_print_composition_v1.json"),
        _load("u2_2a_sensitometry_primitive_v1.json"),
        _load("u5_r2e0_density_domain_operator_v1.json"),
    )


def test_identity_residual_exactly_preserves_base() -> None:
    config, base_config, parent, print_config = _inputs()
    base = build_composition(base_config, parent, print_config)
    operator = SensitometryResidualLUTOperator(base, identity_lut(17), LUTConstraintSpec(**config["constraints"]))
    values = np.random.default_rng(5).random((1000, 3))
    assert np.array_equal(operator.apply(values), base.apply(values))


def test_residual_schema_and_guards_fail_closed() -> None:
    config, base_config, parent, print_config = _inputs()
    base = build_composition(base_config, parent, print_config)
    spec = LUTConstraintSpec(**config["constraints"])
    with pytest.raises(ValueError, match="violates"):
        SensitometryResidualLUTOperator(base, _cyclic_lut(17, 0.1), spec)
    operator = SensitometryResidualLUTOperator(base, identity_lut(17), spec)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        operator.apply(np.array([[1.01, 0.5, 0.5]]))
    payload = operator.to_dict()
    payload["schema"] = "tampered"
    with pytest.raises(ValueError, match="unsupported"):
        SensitometryResidualLUTOperator.from_dict(payload)


def test_frozen_u2_3a_report_is_repeat_identical_and_closes_witness() -> None:
    inputs = _inputs()
    first, first_arrays = evaluate_composition(*inputs)
    second, second_arrays = evaluate_composition(*inputs)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert all(np.array_equal(first_arrays[key], second_arrays[key]) for key in first_arrays)
    assert first["decision"] == "smooth_residual_composition_fail"
    assert not first["checks"]["lut_constraints"]
    assert first["checks"]["identity_composition"]
