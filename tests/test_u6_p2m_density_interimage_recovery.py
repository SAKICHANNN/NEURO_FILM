from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_interimage_recovery import (
    evaluate_interimage_recovery,
    load_contract,
)
from src.film_physics.interimage_development import (
    InterimageDevelopmentOperator,
    independent_development_operator,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p2m_density_interimage_recovery_v1.json"


def _operator() -> InterimageDevelopmentOperator:
    return InterimageDevelopmentOperator(
        (0.1, 0.1, 0.1),
        (2.5, 2.5, 2.5),
        (1.3, 1.3, 1.3),
        (0.0, 0.0, 0.0),
        ((0.0, 0.2, 0.1), (0.1, 0.0, 0.2), (0.2, 0.1, 0.0)),
    )


def test_operator_is_bounded_monotone_and_cross_suppressing() -> None:
    operator = _operator()
    rng = np.random.default_rng(71)
    exposure = rng.uniform(-3.0, 3.0, size=(2048, 3))
    density = operator.apply_log2_exposure(exposure)
    jacobian = operator.derivative_log2_exposure(exposure)
    assert np.all(density > np.asarray(operator.density_min))
    assert np.all(density < np.asarray(operator.density_max))
    assert all(np.all(jacobian[:, index, index] > 0.0) for index in range(3))
    assert all(
        np.all(jacobian[:, row, column] <= 0.0)
        for row in range(3)
        for column in range(3)
        if row != column
    )


def test_zero_coupling_is_exact_independent_curve() -> None:
    operator = _operator()
    independent = independent_development_operator(operator)
    exposure = np.linspace(-3.0, 3.0, 99).reshape(11, 3, 3)
    expected = operator.with_coupling_vector(np.zeros(6)).apply_log2_exposure(
        exposure
    )
    assert np.array_equal(independent.apply_log2_exposure(exposure), expected)


def test_contract_and_evaluator_are_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_interimage_recovery(contract)
    second = evaluate_interimage_recovery(contract)
    assert first == second
    assert first["stable_evidence_id"] == second["stable_evidence_id"]
    assert len(first["witnesses"]) == 3


def test_invalid_coupling_fails_closed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        InterimageDevelopmentOperator(
            (0.1, 0.1, 0.1),
            (2.5, 2.5, 2.5),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
            ((0.0, -0.1, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )


def test_contract_json_remains_strict_json() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["node"] == "U6.P2M"
    assert payload["forbidden_fallbacks"]
