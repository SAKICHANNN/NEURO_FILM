import ast
import inspect
import json
from pathlib import Path

import numpy as np

from src.eval import tst_reference_kernel as k


def small_problem():
    rng = np.random.default_rng(20260908)
    source = rng.random((180, 3))
    targets = [np.clip(source**.8, 0, 1), np.clip(source**1.3, 0, 1)]
    other = rng.random((160, 3))
    other_y = .15+.7*other
    groups = [k.sufficient(source, targets, [0, 1], 3), k.sufficient(other, [other_y], [2], 3)]
    return rng, [(source, targets), (other, [other_y])], groups


def test_observed_objective_hessian_dense_cg():
    rng, pixels, groups = small_problem()
    phi = rng.normal(size=(3, 3))
    reg = k.regularizer(3)
    beta = 2*1e-6/(3*27)
    w = rng.normal(size=(3, 27, 3))*.03
    u = np.einsum("nr,rdc->ndc", phi, w)
    direct = 0.
    constant = 0.
    for (source, targets), group in zip(pixels, groups, strict=True):
        b = k.frozen.lattice_design(source, 3)
        errors = np.stack([source+b@u[i]-y for i, y in zip(group["rows"], targets, strict=True)])
        original = np.stack([source-y for y in targets])
        c = k.pair_matrix(len(targets))
        direct += np.einsum("ipc,ij,jpc->", errors, c, errors)/(len(source)*3)
        constant += np.einsum("ipc,ij,jpc->", original, c, original)/(len(source)*3)
        direct += sum(np.sum(u[i]*(reg@u[i])) for i in group["rows"])/(len(targets)*3)
    direct = direct/2 + beta*np.sum(w*w)/(2*3)
    statistic = k.objective_from_statistics(u, groups, reg, w, beta)["normalized_without_constant"]
    np.testing.assert_allclose(direct-constant/2, statistic, rtol=1e-11, atol=1e-13)
    operator, preconditioner, rhs, diagonal = k.normal_system(phi, groups, reg, beta)
    dense = np.column_stack([operator@v for v in np.eye(81)])
    np.testing.assert_allclose(dense, dense.T, atol=1e-15)
    np.testing.assert_allclose(np.diag(dense), diagonal.ravel(), rtol=1e-12)
    assert np.linalg.eigvalsh(dense).min() > 0
    direction = rng.normal(size=w.shape)
    epsilon = 1e-6
    def objective(weights):
        return k.objective_from_statistics(np.einsum("nr,rdc->ndc", phi, weights), groups, reg, weights, beta)["normalized_without_constant"]
    finite_difference = (objective(w+epsilon*direction)-objective(w-epsilon*direction))/(2*epsilon)
    gradient = np.stack([(operator@w[:, :, c].ravel()).reshape(3, 27)-rhs[:, :, c] for c in range(3)], -1)/3
    np.testing.assert_allclose(finite_difference, np.sum(gradient*direction), rtol=1e-7, atol=1e-9)
    value, state = k.solve_channel(operator, preconditioner, rhs[:, :, 0])
    assert state["solved"]
    np.testing.assert_allclose(value.ravel(), np.linalg.solve(dense, rhs[:, :, 0].ravel()), rtol=1e-5, atol=1e-6)


def test_kernel_rank_and_source_only_swap():
    source = np.array([[0., 0.], [0., 0.], [1., 2.], [1., 2.]])
    reference = np.array([[0.], [1.], [2.], [3.]])
    model, phi = k.kernel_features(source, reference)
    np.testing.assert_allclose(k.predict_features(model, source, reference), phi, atol=1e-12)
    np.testing.assert_allclose(phi@phi.T, (1+k.rbf(source, source, model["source_median"]))*(1+k.rbf(reference, reference, model["reference_median"]))/4, atol=1e-12)
    model["weights"] = np.random.default_rng(48).normal(size=(model["rank"], 27, 3))
    original = k.predict_operators(model, source[:2], reference[:2])
    swapped = k.predict_operators(model, source[:2], reference[:2][::-1])
    np.testing.assert_array_equal(original, swapped[::-1])
    a, features = k.kernel_features(source, None)
    assert a["rank"] == 2
    a["weights"] = np.ones((a["rank"], 27, 3))
    predicted = k.predict_operators(a, source[:2], None)
    np.testing.assert_array_equal(predicted[0], predicted[1])
    np.testing.assert_allclose(k.predict_features(a, source, None), features, atol=1e-12)


def test_freegrid_quadratic_dense_and_zero_rhs():
    _, _, groups = small_problem()
    operator, preconditioner, rhs, _ = k.normal_system(np.eye(3), groups, k.regularizer(3), 0.)
    dense = np.column_stack([operator@v for v in np.eye(81)])
    for channel in range(3):
        value, state = k.solve_channel(operator, preconditioner, rhs[:, :, channel])
        assert state["solved"]
        np.testing.assert_allclose(value.ravel(), np.linalg.solve(dense, rhs[:, :, channel].ravel()), rtol=1e-5, atol=1e-6)
    value, state = k.solve_channel(operator, preconditioner, np.zeros((3, 27)))
    assert state["solved"] and state["iterations"] == 0
    np.testing.assert_array_equal(value, 0.)


def test_descriptor_only_api_and_single_budget_scope():
    assert list(inspect.signature(k.predict_operators).parameters) == ["model", "source", "reference"]
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root/"scripts/run_tst_reference_kernel.py").read_text())
    launches = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "launch"]
    assert len(launches) == 1 and ast.literal_eval(launches[0].args[0]) == "experiment"
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("launch", "Popen") for n in ast.walk(worker))
    config = json.loads((root/"configs/tst_reference_kernel_v1.json").read_text())
    assert config["cpu_seconds"] == config["wall_seconds_per_stage"] == 3600
    assert config["threads"] == 2 and config["rss_bytes"] == 4*1024**3
