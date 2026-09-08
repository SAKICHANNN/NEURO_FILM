from itertools import product

import numpy as np
import pytest

from src.eval.trilinear_lut_certificate import (
    cell_vertex_jacobians,
    numerical_cell_bounds,
)


def identity_cube(size):
    axis = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((r, g, b), axis=-1)


def analyze(cube):
    return numerical_cell_bounds(cube, absolute_margin=1e-12, relative_margin=1e-12)


def nonlinear_values(rgb):
    r, g, b = np.moveaxis(rgb, -1, 0)
    return np.stack((r + 0.7 * r * g * b, g + 0.2 * r * b, b + 0.1 * r * g), -1)


def nonlinear_jacobian(rgb):
    r, g, b = rgb
    return np.array(
        [
            [1 + 0.7 * g * b, 0.7 * r * b, 0.7 * r * g],
            [0.2 * b, 1.0, 0.2 * r],
            [0.1 * g, 0.1 * r, 1.0],
        ]
    )


@pytest.mark.parametrize("size", [2, 5, 16])
def test_identity_all_cells_and_numeric_claim(size):
    cube = identity_cube(size)
    jac = cell_vertex_jacobians(cube)
    np.testing.assert_allclose(jac, np.broadcast_to(np.eye(3), jac.shape), atol=3e-15)
    bounds, report = analyze(cube)
    assert report["cell_count"] == (size - 1) ** 3
    assert report["one_sided_vertex_count"] == 8 * (size - 1) ** 3
    assert bounds.shape == (size - 1, size - 1, size - 1, 5)
    np.testing.assert_allclose(bounds[..., 0], 1.0, atol=3e-15)
    assert report["global_singular_lower_sufficient_numerical_estimate"] > 0.99999999
    assert (
        report["global_residual_condition"]
        == "NUMERICALLY_SATISFIED_SUFFICIENT_CONDITION"
    )
    assert report["rounding_error_certified"] is False
    assert report["photographic_safety_or_quality_claim"] is False


def test_affine_and_bgr_storage_to_rgb_matrix_columns():
    matrix = np.array([[0.8, 0.17, 0.03], [-0.12, 0.71, 0.22], [0.09, -0.11, 0.63]])
    cube = identity_cube(5) @ matrix.T + np.array([0.11, 0.23, -0.07])
    jac = cell_vertex_jacobians(cube)
    np.testing.assert_allclose(jac, np.broadcast_to(matrix, jac.shape), atol=2e-15)
    _, report = analyze(cube)
    assert report["maximum_vertex_spectral_norm"] == pytest.approx(
        np.linalg.norm(matrix, 2)
    )
    assert report["maximum_vertex_residual_spectral_norm"] == pytest.approx(
        np.linalg.norm(matrix - np.eye(3), 2)
    )


def test_nonlinear_interior_jacobian_is_corner_convex_combination():
    size = 4
    cube = nonlinear_values(identity_cube(size))
    vertices = cell_vertex_jacobians(cube)
    bounds, _ = analyze(cube)
    rng = np.random.default_rng(64)
    for b, g, r in product(range(size - 1), repeat=3):
        for _ in range(8):
            frac_bgr = rng.uniform(0.01, 0.99, 3)
            exact = nonlinear_jacobian(
                (np.array([r, g, b]) + frac_bgr[::-1]) / (size - 1)
            )
            weights = np.array(
                [
                    np.prod(np.where(corner, frac_bgr, 1.0 - frac_bgr))
                    for corner in product((0, 1), repeat=3)
                ]
            )
            interpolated = np.einsum("v,vij->ij", weights, vertices[b, g, r])
            np.testing.assert_allclose(interpolated, exact, atol=2e-15)
            assert np.linalg.norm(exact, 2) <= bounds[b, g, r, 2]
            assert np.linalg.norm(exact - np.eye(3), 2) <= bounds[b, g, r, 3]
            assert np.linalg.svd(exact, compute_uv=False)[-1] >= bounds[b, g, r, 4]


def test_shared_grid_face_has_two_one_sided_derivatives():
    cube = identity_cube(3)
    cube[:, :, 1, 0] = 0.2
    jac = cell_vertex_jacobians(cube)
    # BGR corner 001 is the upper-R vertex, while 000 is lower-R.
    assert jac[0, 0, 0, 1, 0, 0] == pytest.approx(0.4)
    assert jac[0, 0, 1, 0, 0, 0] == pytest.approx(1.6)
    assert analyze(cube)[1]["maximum_vertex_spectral_norm"] == pytest.approx(1.6)


def test_sparse_image_colour_samples_can_miss_local_gain_and_reversal():
    from scripts.diagnose_ai_vcg_stage_gain import jacobians

    cube = identity_cube(16)
    cube[:, :, 7, 0] += 0.3
    sample_rgb = np.array([[0.1, 0.15, 0.75], [0.9, 0.8, 0.4]])
    sampled = jacobians(cube, sample_rgb)
    np.testing.assert_allclose(
        sampled, np.broadcast_to(np.eye(3), sampled.shape), atol=2e-12
    )
    _, report = analyze(cube)
    assert report["maximum_vertex_spectral_norm"] == pytest.approx(5.5)
    assert report["vertex_determinant_samples_only"]["negative_count"] > 0
    assert (
        report["vertex_determinant_samples_only"]["full_cell_sign_certified"] is False
    )


def test_analytic_jacobian_crosschecks_existing_finite_difference_inside_cells():
    from scripts.diagnose_ai_vcg_stage_gain import jacobians

    cube = nonlinear_values(identity_cube(16))
    rgb = np.array([[0.12, 0.27, 0.43], [0.78, 0.52, 0.94]])
    finite_difference = jacobians(cube, rgb)
    expected = np.stack([nonlinear_jacobian(point) for point in rgb])
    np.testing.assert_allclose(finite_difference, expected, atol=3e-12)


def test_inconclusive_sufficient_condition_does_not_reject_invertible_axis_rotation():
    rotation = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    _, report = analyze(identity_cube(4) @ rotation.T)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert report["maximum_vertex_spectral_norm"] == pytest.approx(1.0)
    assert report["global_residual_condition"] == "INCONCLUSIVE"
    assert report["global_singular_lower_sufficient_numerical_estimate"] == 0.0


def test_node_clipping_is_different_from_clipping_interpolated_outputs():
    raw = identity_cube(2)
    raw[..., 0] = raw[..., 0] * 2.0 - 0.5
    clipped = np.clip(raw, 0, 1)
    assert analyze(raw)[1]["maximum_vertex_spectral_norm"] == pytest.approx(2.0)
    assert analyze(clipped)[1]["maximum_vertex_spectral_norm"] == pytest.approx(1.0)
    r = 0.4
    output_clipped_r = np.clip((1 - r) * raw[0, 0, 0, 0] + r * raw[0, 0, 1, 0], 0, 1)
    node_clipped_r = (1 - r) * clipped[0, 0, 0, 0] + r * clipped[0, 0, 1, 0]
    assert output_clipped_r == pytest.approx(0.3)
    assert node_clipped_r == pytest.approx(0.4)


@pytest.mark.parametrize(
    "cube",
    [
        np.zeros((2, 3, 2, 3)),
        np.zeros((1, 1, 1, 3)),
        np.zeros((2, 2, 2, 3), int),
        np.full((2, 2, 2, 3), np.nan),
        np.zeros((2, 2, 2, 4)),
    ],
)
def test_invalid_cube_rejected(cube):
    with pytest.raises(ValueError, match="finite floating"):
        cell_vertex_jacobians(cube)


@pytest.mark.parametrize("margin", [-1e-12, np.inf, np.nan])
def test_invalid_numerical_margin_rejected(margin):
    with pytest.raises(ValueError, match="numerical margins"):
        numerical_cell_bounds(
            identity_cube(2), absolute_margin=margin, relative_margin=0
        )


def test_frozen_sources_fail_on_hash_drift_without_output(tmp_path):
    from scripts.audit_vcg_lut_certificate import read_bound

    source = tmp_path / "source.json"
    source.write_bytes(b"{}")
    with pytest.raises(ValueError, match="Source hash drift"):
        read_bound(source, "0" * 64)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source.json"]
