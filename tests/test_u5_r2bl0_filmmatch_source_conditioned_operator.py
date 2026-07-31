import numpy as np

from src.eval.filmmatch_source_conditioned_operator import (
    decode_operator_parameters,
    encode_operator_parameters,
    fit_ridge,
    predict_ridge,
    select_nested_alpha,
    source_descriptor,
)


def _descriptor_config() -> dict:
    return {
        "source_descriptor": {
            "rgb_quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
            "luma_quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
            "chromaticity_quantiles": [0.1, 0.5, 0.9],
            "chroma_quantiles": [0.1, 0.5, 0.9],
            "epsilon": 1e-8,
        }
    }


def test_operator_parameter_roundtrip_is_exact_enough() -> None:
    parameters = np.linspace(-2.0, 1.0, 12)
    operator, clipped = decode_operator_parameters(
        parameters,
        curve_identity_mixture=0.25,
        matrix_identity_mixture=0.25,
        free_logit_bounds=(-8.0, 4.0),
    )
    recovered = encode_operator_parameters(
        operator,
        curve_identity_mixture=0.25,
        matrix_identity_mixture=0.25,
    )
    assert clipped == 0
    np.testing.assert_allclose(recovered, parameters, atol=1e-12, rtol=0.0)


def test_decode_clips_and_remains_structurally_safe() -> None:
    operator, clipped = decode_operator_parameters(
        np.asarray([-20.0, 20.0] * 6),
        curve_identity_mixture=0.25,
        matrix_identity_mixture=0.25,
        free_logit_bounds=(-8.0, 4.0),
    )
    cube = np.asarray([[0.0, 0.0, 0.0], [0.3, 0.5, 0.7], [1.0, 1.0, 1.0]])
    output = operator.apply(cube)
    assert clipped == 12
    assert np.all((output >= 0.0) & (output <= 1.0))
    assert np.all(operator.jacobian_determinants(cube) > 0.0)


def test_source_descriptor_is_finite_and_target_free() -> None:
    source = np.linspace(0.0, 1.0, 48, dtype=np.float64).reshape(16, 3)
    descriptor = source_descriptor(source, _descriptor_config())
    assert descriptor.shape == (29,)
    assert np.all(np.isfinite(descriptor))


def test_ridge_and_nested_selection_use_group_held_rows() -> None:
    x = np.asarray(
        [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]
    )
    y = np.column_stack((2.0 * x[:, 0] + 1.0, -x[:, 1] + 0.25))
    labels = np.asarray(["a", "a", "b", "b"])
    alpha, rows = select_nested_alpha(x, y, labels, [0.1, 1.0, 10.0])
    model = fit_ridge(x, y, alpha=alpha)
    predicted = predict_ridge(model, x)
    assert len(rows) == 3
    assert alpha in {0.1, 1.0, 10.0}
    assert predicted.shape == y.shape
    assert np.all(np.isfinite(predicted))
