from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.digital_lad_print_compatibility import _build_print_operator
from src.eval.digital_lad_print_sensitivity import load_contract, run_audit
from src.film_physics.print_sensitivity import (
    apply_density_to_print_with_jacobian,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6t_digital_lad_print_sensitivity_v1.json"


def _operator():
    p6s = json.loads(
        (ROOT / "configs/u6_p6s_digital_lad_print_compatibility_v1.json").read_text(
            encoding="utf-8"
        )
    )
    sensitometry = json.loads(
        (ROOT / p6s["parents"]["sensitometry_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    source = json.loads(
        (ROOT / p6s["parents"]["print_source_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    return _build_print_operator(sensitometry, source, p6s)


def test_analytic_jacobian_matches_finite_difference() -> None:
    operator = _operator()
    density = np.asarray([[0.89, 0.89, 0.89], [1.04, 1.04, 1.04]])
    output, jacobian = apply_density_to_print_with_jacobian(operator, density)
    assert np.array_equal(output, operator.apply(density))
    step = 1e-6
    for channel in range(3):
        offset = np.zeros_like(density)
        offset[:, channel] = step
        finite = (operator.apply(density + offset) - operator.apply(density - offset)) / (
            2.0 * step
        )
        assert np.max(np.abs(finite - jacobian[:, :, channel])) <= 2e-8


def test_sensitivity_rejects_invalid_operator_and_density() -> None:
    with pytest.raises(TypeError, match="DensityToPrintInterpretation"):
        apply_density_to_print_with_jacobian(object(), np.ones((1, 3)))
    with pytest.raises(ValueError, match="outside declared references"):
        apply_density_to_print_with_jacobian(_operator(), np.zeros((1, 3)))


def test_formal_audit_obeys_frozen_contract() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert set(report["gate_results"]) == set(
        load_contract(CONTRACT)["automatic_gates"]
    )
    assert report["code_counts"] == {"negative": 824, "interpositive": 824}
    assert report["gate_results"]["rgb_image_transform_count_zero"]
