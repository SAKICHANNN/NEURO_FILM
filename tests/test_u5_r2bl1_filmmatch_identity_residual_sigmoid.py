import numpy as np

from src.roll2film.identity_residual_sigmoid import (
    IdentityResidualSigmoidOperator,
    fit_identity_residual_sigmoid,
)


def _operator() -> IdentityResidualSigmoidOperator:
    return IdentityResidualSigmoidOperator(
        capture_matrix=np.asarray(
            [[0.9, 0.07, 0.03], [0.04, 0.9, 0.06], [0.02, 0.08, 0.9]]
        ),
        response_midpoints=np.asarray([-3.0, -2.5, -2.0]),
        response_slopes=np.asarray([1.2, 1.0, 0.8]),
        scan_matrix=np.asarray(
            [[0.92, 0.05, 0.03], [0.03, 0.92, 0.05], [0.04, 0.04, 0.92]]
        ),
        nonlinear_strength=0.75,
    )


def test_operator_preserves_cube_endpoints_and_has_derivative_floor() -> None:
    operator = _operator()
    cube = np.stack(
        np.meshgrid(*(np.linspace(0.0, 1.0, 9),) * 3, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    output = operator.apply(cube)
    np.testing.assert_allclose(output[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(output[-1], 1.0, atol=1e-12)
    assert np.all((output >= 0.0) & (output <= 1.0))
    assert float(np.min(operator.jacobian_determinants(cube))) > 0.0009


def test_operator_partition_is_exact() -> None:
    rng = np.random.default_rng(7)
    source = rng.random((64, 3))
    operator = _operator()
    full = operator.apply(source)
    tiled = np.concatenate([operator.apply(source[:17]), operator.apply(source[17:])])
    np.testing.assert_array_equal(full, tiled)


def test_fit_improves_identity_on_synthetic_operator() -> None:
    rng = np.random.default_rng(9)
    source = rng.random((256, 3))
    target = _operator().apply(source)
    fit = fit_identity_residual_sigmoid(
        source,
        target,
        restart_count=1,
        maximum_function_evaluations=800,
        seed=11,
    )
    identity_rmse = float(np.sqrt(np.mean(np.square(source - target))))
    assert fit.converged
    assert fit.development_rgb_rmse < 0.25 * identity_rmse
    assert np.min(fit.operator.jacobian_determinants(source)) > 0.0009
