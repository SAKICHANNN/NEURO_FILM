import numpy as np
from pillow_lut import identity_table

from scripts.diagnose_ai_vcg_stage_gain import describe, interpolate, jacobians


def exact_identity_cube():
    axis = np.linspace(0, 1, 16, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij")[::-1], axis=-1)


def test_identity_values_derivatives_and_endpoints():
    cube = exact_identity_cube()
    points = np.array([[0, 0, 0], [1, 1, 1], [0.23, 0.47, 0.82]])
    np.testing.assert_allclose(interpolate(cube, points), points, atol=1e-14)
    jac = jacobians(cube, points)
    np.testing.assert_allclose(jac, np.broadcast_to(np.eye(3), jac.shape), atol=1e-10)
    assert describe(jac)["negative_determinant_fraction"] == 0


def test_linear_matrix_orientation_and_composition():
    cube = exact_identity_cube()
    matrix = np.array([[0.8, 0.1, 0], [0.1, 0.7, 0.2], [0, 0, 0.5]])
    transformed = cube @ matrix.T
    points = np.random.default_rng(1).uniform(0.1, 0.9, (23, 3))
    np.testing.assert_allclose(interpolate(transformed, points), points @ matrix.T)
    jac = jacobians(transformed, points)
    np.testing.assert_allclose(jac, np.broadcast_to(matrix, jac.shape), atol=1e-10)
    swap = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert describe(jac @ swap)["negative_determinant_fraction"] == 1


def test_pillow_quantized_node_values_preserved():
    cube = np.asarray(identity_table(16).table).reshape(16, 16, 16, 3)
    points = exact_identity_cube().reshape(-1, 3)
    np.testing.assert_allclose(
        interpolate(cube, points), cube.reshape(-1, 3), atol=1e-14
    )


def test_official_ncc_preserves_source_without_covariance():
    from scripts.diagnose_ai_vcg_stage_gain import (
        REPORT,
        REPORT_SHA,
        official_functions,
    )
    from scripts.run_ai_vcg_reference import checked_json

    scope = official_functions(checked_json(REPORT, REPORT_SHA))
    source = np.zeros((1, 19, 17, 3), dtype=np.uint8)
    source[0, ..., 0] = 123
    ref = np.ones((20, 20, 3), dtype=np.uint8)

    def forbidden(*args):
        raise AssertionError("ncc must not fit covariance")

    scope["vars"] = forbidden
    full, small = scope["preprocess"](source, ref, 32, True)
    assert np.array_equal(full, source)
    assert np.array(small).shape == (1, 32, 32, 3)
