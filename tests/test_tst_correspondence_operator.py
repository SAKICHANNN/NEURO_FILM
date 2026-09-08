import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import lsq_linear

from src.eval import tst_correspondence_operator as core
from src.eval import tst_reference_response as frozen


def config():
    return json.loads(Path('C:/Users/hhvrf/Documents/neuro_film/configs/tst_correspondence_operator_feasibility_v1.json').read_text())


def test_exact_objective_gradient_and_absolute_residual_equivalence():
    rng = np.random.default_rng(48)
    x, y = rng.random((91, 3)), rng.random((91, 3))
    v = rng.random((27, 3))
    ident = core.identity_grid(3)
    d2 = frozen.second_derivatives(3)
    matrix, rhs = core.augmented_system(x, y, 3, 1e-6, 1e-8)
    loss, gradient = core.objective_and_gradient(v, matrix, rhs)
    direct = np.mean((frozen.lattice_design(x, 3)@v-y)**2)+1e-6*np.mean((d2@(v-ident))**2)+1e-8*np.mean((v-ident)**2)
    np.testing.assert_allclose(loss, direct, rtol=1e-13, atol=1e-15)
    direction = rng.normal(size=v.shape)
    eps = 1e-6
    plus = core.objective_and_gradient(v+eps*direction, matrix, rhs)[0]
    minus = core.objective_and_gradient(v-eps*direction, matrix, rhs)[0]
    np.testing.assert_allclose((plus-minus)/(2*eps), np.sum(gradient*direction), rtol=1e-8, atol=1e-10)
    basis = frozen.lattice_design(x, 3)
    np.testing.assert_allclose(basis@ident, x, atol=1e-15)
    actual = core.render_absolute(x, v, 3)
    np.testing.assert_allclose(actual, frozen.render(x, v-ident, 3), atol=1e-15)
    axis = np.linspace(0, 1, 3)
    independent = RegularGridInterpolator((axis, axis, axis), v.reshape(3, 3, 3, 3))(x)
    np.testing.assert_array_equal(np.rint(actual*65535).astype(np.uint16), np.rint(independent*65535).astype(np.uint16))


def test_active_bound_kkt_and_wrong_sign_rejection():
    matrix = np.eye(3)
    rhs = np.array([-.2, .4, 1.2])
    solution = lsq_linear(matrix, rhs, bounds=(0, 1), tol=1e-14)
    expected = np.clip(rhs, 0, 1)
    np.testing.assert_allclose(solution.x, expected, atol=1e-12)
    gradient = 2*(expected-rhs)/3
    result = core.kkt_audit(expected, gradient, 1., 1e-10, 1e-8, 1e-12)
    assert result['passed'] and result['lower_active'] == result['upper_active'] == 1
    assert not core.kkt_audit(expected, -gradient, 1., 1e-10, 1e-8, 1e-12)['passed']
    assert not core.kkt_audit(expected+.1, gradient, 1., 1e-10, 1e-8, 1e-12)['passed']


def test_solver_known_identity_and_nonlinear_dense_reference():
    cfg = {**config(), 'dimension': 3}
    source = core.identity_grid(3)
    identity, identity_state = core.fit_operator(source, source, cfg)
    assert identity_state['solved']
    np.testing.assert_allclose(identity, source, atol=1e-8)
    target = .1+.8*source**1.4
    value, state = core.fit_operator(source, target, cfg)
    assert state['solved']
    a, b = core.augmented_system(source, target, 3, cfg['smoothness'], cfg['tie'])
    exact = np.linalg.solve((a.T@a).toarray(), a.T@b)
    assert exact.min() > 0 and exact.max() < 1
    np.testing.assert_allclose(value, exact, rtol=1e-8, atol=1e-9)
    assert state['objective'] >= 0


def test_support_and_frozen_queue_have_no_new_or_development_inputs():
    train = np.array([[0., 0., 0.], [.01, .01, .01]])
    support = core.support_summary(train, np.array([[1., 1., 1.]]), 7)
    assert support['observed_nodes'] == 8
    assert support['mean_unobserved_node_interpolation_weight'] == 1.
    cfg = config()
    assert len(cfg['pairs']) == 8 and len(cfg['probes']) == 4
    assert all(r['group'] < 24 for r in cfg['pairs'])
    assert cfg['dimension'] == 7 and cfg['smoothness'] == 1e-6 and cfg['tie'] == 1e-8
    assert cfg['cpu_seconds'] == cfg['wall_seconds_per_stage'] == 600


def test_reduced_bounded_solve_matches_original_dense_system():
    cfg = {**config(), 'dimension': 3}
    rng = np.random.default_rng(123)
    source = rng.random((45, 3))
    target = (source > .5).astype(float)
    actual, state = core.fit_operator(source, target, cfg)
    assert state['solved']
    matrix, rhs = core.augmented_system(source, target, 3, cfg['smoothness'], cfg['tie'])
    expected = np.stack([lsq_linear(matrix.toarray(), rhs[:, c], bounds=(0, 1),
                        method='bvls', tol=cfg['solver']['tol'], max_iter=cfg['solver']['max_iter']).x
                        for c in range(3)], -1)
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-8)
    assert np.any(actual == 0) and np.any(actual == 1)
    direct, _ = core.objective_and_gradient(expected, matrix, rhs)
    np.testing.assert_allclose(state['objective'], direct, atol=1e-13)
