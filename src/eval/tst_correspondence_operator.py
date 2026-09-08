import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
from scipy.optimize import lsq_linear

from src.eval import tst_reference_response as frozen


def identity_grid(dimension: int) -> np.ndarray:
    axis = np.linspace(0., 1., dimension)
    return np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)


def augmented_system(source, target, dimension, smoothness, tie):
    assert source.shape == target.shape and source.ndim == 2 and source.shape[1] == 3
    assert np.isfinite(target).all() and target.min() >= 0 and target.max() <= 1
    identity = identity_grid(dimension)
    design = frozen.lattice_design(source, dimension)
    second = frozen.second_derivatives(dimension)
    matrix = sparse.vstack([design/np.sqrt(len(source)),
                            np.sqrt(smoothness/second.shape[0])*second,
                            np.sqrt(tie/dimension**3)*sparse.eye(dimension**3)], format='csr')
    rhs = np.vstack([target/np.sqrt(len(source)),
                     np.sqrt(smoothness/second.shape[0])*(second@identity),
                     np.sqrt(tie/dimension**3)*identity])
    return matrix, rhs


def objective_and_gradient(values, matrix, rhs):
    residual = matrix@values-rhs
    return float(np.sum(residual**2)/3), np.asarray(2*(matrix.T@residual)/3)


def kkt_audit(values, gradient, gradient_scale, absolute_tol, relative_tol, bound_tol):
    projected = values-np.clip(values-gradient, 0., 1.)
    absolute = float(np.max(np.abs(projected)))
    relative = absolute/max(float(gradient_scale), np.finfo(float).eps)
    feasible = bool(np.isfinite(values).all() and values.min() >= -bound_tol and values.max() <= 1+bound_tol)
    return {'projected_gradient_infinity': absolute, 'relative_projected_gradient': relative,
            'gradient_scale': float(gradient_scale), 'bound_feasible': feasible,
            'lower_active': int(np.sum(values <= bound_tol)), 'upper_active': int(np.sum(values >= 1-bound_tol)),
            'passed': bool(feasible and absolute <= absolute_tol and relative <= relative_tol)}


def fit_operator(source, target, config):
    d, solver = config['dimension'], config['solver']
    matrix, rhs = augmented_system(source, target, d, config['smoothness'], config['tie'])
    identity = identity_grid(d)
    gram = (matrix.T@matrix).toarray()
    square_root = np.linalg.cholesky(gram).T
    reduced_rhs = solve_triangular(square_root.T, matrix.T@rhs, lower=True)
    values, states = [], []
    for channel in range(3):
        solution = lsq_linear(square_root, reduced_rhs[:, channel], bounds=(0., 1.), method='bvls',
                              tol=solver['tol'], lsq_solver='exact', max_iter=solver['max_iter'])
        value = solution.x
        gradient = np.asarray(2*(matrix.T@(matrix@value-rhs[:, channel]))/3)
        initial_gradient = np.asarray(2*(matrix.T@(matrix@identity[:, channel]-rhs[:, channel]))/3)
        scale = max(np.max(np.abs(initial_gradient)), solver['gradient_scale_floor'])
        audit = kkt_audit(value, gradient, scale, solver['kkt_absolute'], solver['kkt_relative'], solver['bound_tol'])
        states.append({'channel': channel, 'status': int(solution.status), 'iterations': int(solution.nit),
                       'solver_success': bool(solution.success), 'reported_optimality': float(solution.optimality),
                       **audit, 'solved': bool(solution.success and audit['passed'])})
        values.append(value)
    grid = np.stack(values, -1)
    loss, _ = objective_and_gradient(grid, matrix, rhs)
    return grid, {'channels': states, 'solved': all(r['solved'] for r in states),
                  'objective': loss, 'matrix_shape': list(matrix.shape), 'matrix_nnz': int(matrix.nnz)}


def render_absolute(source, values, dimension=7):
    return frozen.lattice_design(source, dimension)@values


def support_summary(fit_colors, scoring_colors, dimension):
    design = frozen.lattice_design(fit_colors, dimension)
    support = np.asarray(design.sum(0)).ravel() > 0
    scoring = frozen.lattice_design(scoring_colors, dimension)
    weight = np.asarray(scoring[:, ~support].sum(1)).ravel()
    return {'observed_nodes': int(support.sum()), 'nodes': dimension**3,
            'scoring_pixels_touching_unobserved_nodes_fraction': float(np.mean(weight > 1e-12)),
            'mean_unobserved_node_interpolation_weight': float(weight.mean()),
            'claim': 'Node support diagnostic only; outside observed donor colors is a regularized extension, not recovered treatment truth.'}
