from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_langmuir_interimage import (
    evaluate_langmuir_interimage,
    load_contract,
)
from src.film_physics.interimage_development import InterimageDevelopmentOperator
from src.film_physics.langmuir_interimage import (
    LangmuirDonorProfile,
    apply_langmuir_interimage_development,
    langmuir_interimage_jacobian,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2v_langmuir_interimage_primitive_v1.json"


def _operator() -> InterimageDevelopmentOperator:
    return InterimageDevelopmentOperator(
        (0.08, 0.08, 0.08),
        (2.55, 2.55, 2.55),
        (1.35, 1.35, 1.35),
        (0.0, 0.0, 0.0),
        ((0.0, 0.28, 0.18), (0.18, 0.0, 0.28), (0.28, 0.18, 0.0)),
    )


def test_donor_matches_reference_and_linear_limit() -> None:
    values = np.linspace(0.0, 1.0, 257)[:, None].repeat(3, axis=1)
    profile = LangmuirDonorProfile((1.0,) * 3, (1.0,) * 3, (0.5,) * 3)
    assert np.array_equal(profile.apply(np.full((1, 3), 0.5)), np.full((1, 3), 0.5))
    assert np.all(profile.derivative(values) > 0.0)
    linear = LangmuirDonorProfile((1.0,) * 3, (float("inf"),) * 3, (0.5,) * 3)
    assert np.array_equal(linear.apply(values), values)


def test_operator_jacobian_has_expected_signs() -> None:
    profile = LangmuirDonorProfile((1.0,) * 3, (1.0,) * 3, (0.5,) * 3)
    exposure = np.linspace(-2.0, 2.0, 99).reshape(11, 3, 3)
    output = apply_langmuir_interimage_development(_operator(), exposure, profile)
    jacobian = langmuir_interimage_jacobian(_operator(), exposure, profile)
    assert np.all(np.isfinite(output))
    assert all(np.all(jacobian[..., index, index] > 0.0) for index in range(3))
    assert all(
        np.all(jacobian[..., row, column] <= 0.0)
        for row in range(3)
        for column in range(3)
        if row != column
    )


def test_invalid_donor_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        LangmuirDonorProfile((1.0,) * 3, (1.0,) * 3, (0.0,) * 3)


def test_frozen_evaluator_is_repeat_exact() -> None:
    config = load_contract(CONTRACT)
    first = evaluate_langmuir_interimage(ROOT, config)
    second = evaluate_langmuir_interimage(ROOT, config)
    assert first == second
    assert first["decision"] == "pass-clean-room-primitive"
    assert all(first["gate_checks"].values())


def test_contract_is_strict_json() -> None:
    assert json.loads(CONTRACT.read_text(encoding="utf-8"))["node"].endswith("U6.P2V")
