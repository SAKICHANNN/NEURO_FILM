from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2a_constrained_operator_audit import run_audit
from src.roll2film.constrained import (
    ConstrainedGlobalColorOperator,
    LUTConstraintSpec,
    audit_lut_constraints,
    identity_lut,
)
from src.roll2film.lut import DenseLUT3D, bake_dense_lut
from src.roll2film.operators import AffineColorOperator


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "u5_r2a_constrained_global_operator_v1.json"


def _spec() -> LUTConstraintSpec:
    values = json.loads(CONFIG.read_text(encoding="utf-8"))["lut"]
    return LUTConstraintSpec(
        output_minimum=values["output_minimum"],
        output_maximum=values["output_maximum"],
        maximum_residual_amplitude=values["maximum_residual_amplitude"],
        maximum_first_axis_step=values["maximum_first_axis_step"],
        maximum_second_axis_difference=values["maximum_second_axis_difference"],
        maximum_neutral_axis_error=values["maximum_neutral_axis_error"],
        minimum_tetrahedron_jacobian_determinant=values[
            "minimum_tetrahedron_jacobian_determinant"
        ],
    )


def _scalar_reference(lut: DenseLUT3D, point: np.ndarray) -> np.ndarray:
    coordinate = point * (lut.size - 1)
    lower = np.minimum(np.floor(coordinate).astype(int), lut.size - 2)
    f = coordinate - lower
    r, g, b = lower
    red, green, blue = f
    c000 = lut.values[r, g, b]
    c100 = lut.values[r + 1, g, b]
    c010 = lut.values[r, g + 1, b]
    c001 = lut.values[r, g, b + 1]
    c110 = lut.values[r + 1, g + 1, b]
    c101 = lut.values[r + 1, g, b + 1]
    c011 = lut.values[r, g + 1, b + 1]
    c111 = lut.values[r + 1, g + 1, b + 1]
    if red >= green >= blue:
        return c000 + red * (c100 - c000) + green * (c110 - c100) + blue * (c111 - c110)
    if red >= blue > green:
        return c000 + red * (c100 - c000) + blue * (c101 - c100) + green * (c111 - c101)
    if blue > red >= green:
        return c000 + blue * (c001 - c000) + red * (c101 - c001) + green * (c111 - c101)
    if green > red >= blue:
        return c000 + green * (c010 - c000) + red * (c110 - c010) + blue * (c111 - c110)
    if green >= blue > red:
        return c000 + green * (c010 - c000) + blue * (c011 - c010) + red * (c111 - c011)
    return c000 + blue * (c001 - c000) + green * (c011 - c001) + red * (c111 - c011)


def test_tetrahedral_identity_and_independent_reference() -> None:
    identity = identity_lut(9)
    rng = np.random.default_rng(72)
    probes = rng.uniform(0.0, 1.0, size=(4096, 3))
    assert np.array_equal(identity.apply(probes), probes)

    values = identity.values.copy()
    values[..., 0] += 0.02 * values[..., 1] * (1.0 - values[..., 2])
    lut = DenseLUT3D(values, np.zeros(3), np.ones(3), "tetrahedral")
    expected = np.asarray([_scalar_reference(lut, point) for point in probes])
    assert np.max(np.abs(lut.apply(probes) - expected)) < 1e-12


def test_tetrahedron_jacobians_match_linear_matrix() -> None:
    matrix = np.asarray(
        [
            [0.65, 0.0, 0.35],
            [0.0, 1.0, 0.0],
            [0.0, 0.35, 0.65],
        ]
    )
    lut = bake_dense_lut(
        AffineColorOperator(matrix, np.zeros(3)),
        5,
        interpolation="tetrahedral",
    )
    determinants = lut.tetrahedron_jacobian_determinants()
    assert determinants.shape == (4, 4, 4, 6)
    assert np.max(np.abs(determinants - np.linalg.det(matrix))) < 1e-12


def test_constraint_audit_accepts_counterexample_but_not_semantic_claim() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    matrix = np.asarray(config["non_safety_counterexample"]["matrix"])
    lut = bake_dense_lut(
        AffineColorOperator(matrix, np.zeros(3)),
        config["lut"]["audit_size"],
        interpolation="tetrahedral",
    )
    report = audit_lut_constraints(lut, _spec())
    probe = np.asarray([config["non_safety_counterexample"]["blue_probe"]])
    output = lut.apply(probe)

    assert report.passes
    assert output[0, 0] - probe[0, 0] >= 0.25
    assert report.maximum_neutral_axis_error <= 1e-12
    assert report.minimum_tetrahedron_jacobian_determinant > 0.0


def test_global_operator_replay_nonmutation_and_fail_closed() -> None:
    operator = ConstrainedGlobalColorOperator.identity(5, _spec())
    replay = ConstrainedGlobalColorOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    source = np.random.default_rng(73).uniform(0.0, 1.0, size=(31, 47, 3))
    before = source.copy()

    assert np.array_equal(operator.apply(source), source)
    assert np.array_equal(replay.apply(source), source)
    assert np.array_equal(source, before)
    with pytest.raises(ValueError, match="outside"):
        operator.apply(np.asarray([[1.01, 0.5, 0.5]]))
    with pytest.raises(ValueError, match="finite"):
        operator.apply(np.asarray([[np.nan, 0.5, 0.5]]))


def test_constraint_violation_and_metadata_tamper_fail_closed() -> None:
    identity = identity_lut(5)
    values = identity.values.copy()
    values[0, 0, 0] = [1.0, 0.0, 0.0]
    unsafe = DenseLUT3D(values, np.zeros(3), np.ones(3), "tetrahedral")
    with pytest.raises(ValueError, match="violates"):
        ConstrainedGlobalColorOperator(
            ConstrainedGlobalColorOperator.identity(5, _spec()).base,
            unsafe,
            _spec(),
        )

    payload = ConstrainedGlobalColorOperator.identity(5, _spec()).to_dict()
    payload["working_space"] = "acescg"
    with pytest.raises(ValueError, match="working-space"):
        ConstrainedGlobalColorOperator.from_dict(payload)


def test_frozen_audit_is_repeat_identical_and_passes() -> None:
    first = run_audit(CONFIG)
    second = run_audit(CONFIG)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["decision"] == "pass_numerical_representation_not_semantic_safety"
    assert all(first["automatic_gate"].values())
