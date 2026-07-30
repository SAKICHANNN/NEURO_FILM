from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.filmmatch_root_polynomial import (
    fit_models,
    ordinary_exponents,
    polynomial_features,
    root_exponents,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (
        ROOT / "configs/u5_r2bf0_filmmatch_root_polynomial_baseline_v1.json"
    ).read_text(encoding="utf-8")
)


def test_root_features_deduplicate_exponent_rays() -> None:
    degree2 = root_exponents(2)
    degree3 = root_exponents(3)
    assert len(degree2) == 6
    assert len(degree3) == 13
    assert degree3[: len(degree2)] == degree2
    assert (2, 0, 0) not in degree2
    assert len(ordinary_exponents(3)) == 19


def test_signed_root_features_are_positive_scale_homogeneous() -> None:
    values = np.asarray(
        [[-0.1, 0.2, 0.4], [0.0, -0.3, 1.2], [0.5, 0.7, -0.2]]
    )
    exponents = root_exponents(3)
    base = polynomial_features(values, exponents, root=True)
    for scale in (0.25, 0.5, 2.0, 4.0):
        actual = polynomial_features(values * scale, exponents, root=True)
        np.testing.assert_allclose(actual, base * scale, atol=2e-15, rtol=2e-15)


def test_root_operator_recovers_homogeneous_synthetic_mapping() -> None:
    rng = np.random.default_rng(20260728)
    source = rng.uniform(-0.1, 1.2, size=(500, 3))
    exponents = root_exponents(2)
    features = polynomial_features(source, exponents, root=True)
    coefficients = rng.normal(0.0, 0.2, size=(len(exponents), 3))
    target = features @ coefficients
    fits = fit_models(source, target, CONFIG)
    prediction = fits["signed_root_polynomial_degree2"].apply(source)
    np.testing.assert_allclose(prediction, target, atol=2e-5, rtol=2e-5)


def test_fit_is_deterministic_and_full_affine_can_recover_bias() -> None:
    rng = np.random.default_rng(7)
    source = rng.uniform(0.0, 1.0, size=(100, 3))
    matrix = np.asarray(
        [[1.1, 0.1, -0.05], [0.0, 0.9, 0.1], [0.05, 0.0, 0.8]]
    )
    bias = np.asarray([0.02, -0.01, 0.03])
    target = source @ matrix.T + bias
    first = fit_models(source, target, CONFIG)
    second = fit_models(source, target, CONFIG)
    for name in first:
        np.testing.assert_array_equal(
            first[name].coefficients, second[name].coefficients
        )
        np.testing.assert_array_equal(first[name].bias, second[name].bias)
    np.testing.assert_allclose(
        first["full_affine"].apply(source), target, atol=1e-12, rtol=1e-12
    )
