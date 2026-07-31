import numpy as np

from src.roll2film.identity_residual_sigmoid import IdentityResidualSigmoidOperator
from src.roll2film.strict_interior_sigmoid import (
    StrictInteriorSigmoidOperator,
    fit_strict_interior_sigmoid,
)


def test_strict_interior_operator_excludes_quantized_cube_boundaries() -> None:
    operator = StrictInteriorSigmoidOperator(
        base=IdentityResidualSigmoidOperator(
            capture_matrix=np.eye(3),
            response_midpoints=np.full(3, -3.0),
            response_slopes=np.ones(3),
            scan_matrix=np.eye(3),
        )
    )
    axis = np.linspace(0.0, 1.0, 17)
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    output = operator.apply(cube)
    code = np.rint(output * 65535.0).astype(np.uint16)
    assert np.all(code > 0)
    assert np.all(code < 65535)
    assert np.min(operator.jacobian_determinants(cube)) > 0.0


def test_strict_interior_fit_improves_over_identity() -> None:
    rng = np.random.default_rng(41)
    source = rng.uniform(0.0, 1.0, size=(96, 3))
    truth = StrictInteriorSigmoidOperator(
        base=IdentityResidualSigmoidOperator(
            capture_matrix=np.eye(3),
            response_midpoints=np.asarray([-2.0, -2.5, -3.0]),
            response_slopes=np.asarray([1.2, 1.4, 1.6]),
            scan_matrix=np.eye(3),
        )
    )
    target = truth.apply(source)
    result = fit_strict_interior_sigmoid(
        source, target, restart_count=1, maximum_function_evaluations=500
    )
    fit_rmse = np.sqrt(np.mean(np.square(result.operator.apply(source) - target)))
    identity_rmse = np.sqrt(np.mean(np.square(source - target)))
    assert result.converged
    assert fit_rmse < identity_rmse * 0.1
