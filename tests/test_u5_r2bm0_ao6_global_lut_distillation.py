from __future__ import annotations

import numpy as np

from src.roll2film.global_lut_distillation import (
    MonotoneRGBShaper,
    ShapedGlobalLUT,
    fit_shaped_global_lut,
)


def _fixture(count: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    source = rng.random((count, 3))
    target = np.column_stack(
        (
            0.03 + 0.90 * source[:, 0] ** 0.9,
            0.02 + 0.92 * source[:, 1] ** 1.05,
            0.04 + 0.88 * source[:, 2] ** 0.95,
        )
    )
    return source, target


def test_fit_is_explicit_safe_and_roundtrips() -> None:
    source, target = _fixture()
    model = fit_shaped_global_lut(
        source,
        target,
        shaper_knots=17,
        minimum_shaper_increment=1e-4,
        lut_size=5,
        identity_regularization=0.1,
        difference_regularization=1.0,
        lsqr_iteration_limit=80,
        output_margin=1.0 / 65535.0,
        minimum_determinant=0.01,
        strength_iterations=28,
    )
    output = model.apply(source)
    identity_rmse = float(np.sqrt(np.mean((source - target) ** 2)))
    candidate_rmse = float(np.sqrt(np.mean((output - target) ** 2)))
    assert candidate_rmse < identity_rmse
    assert np.min(output) > 0.0
    assert np.max(output) < 1.0
    assert np.min(model.lut.tetrahedron_jacobian_determinants()) >= 0.01 - 1e-8
    replay = ShapedGlobalLUT.from_dict(model.to_dict())
    np.testing.assert_array_equal(replay.apply(source), output)


def test_shaper_rejects_non_monotone_curves() -> None:
    values = np.linspace(0.0, 1.0, 5)[:, None] * np.ones((1, 3))
    values[2, 0] = values[1, 0]
    try:
        MonotoneRGBShaper(values)
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("non-monotone shaper accepted")


def test_apply_rejects_out_of_domain_rgb() -> None:
    values = np.linspace(0.0, 1.0, 5)[:, None] * np.ones((1, 3))
    shaper = MonotoneRGBShaper(values)
    try:
        shaper.apply(np.asarray([[1.01, 0.5, 0.5]]))
    except ValueError as error:
        assert "[0,1]" in str(error)
    else:
        raise AssertionError("out-of-domain RGB accepted")
