import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "oracle", Path(__file__).resolve().parents[1] / "scripts/run_tst100k_global_oracle.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_known_global_cross_channel_fit_and_axis_boundaries():
    rng = np.random.default_rng(48)
    x = rng.random((4000, 3))
    matrix = np.array([[.55, .08, .04], [.03, .6, .06], [.09, .04, .5]])
    target = x @ matrix + np.array([.1, .12, .15])
    target[:, 0] += .05 * x[:, 0] * x[:, 1]
    lut, states = oracle.fit_lut(x, target, 5, 1e-6)
    check = np.vstack([rng.random((500, 3)), np.eye(3), np.zeros((1, 3)), np.ones((1, 3))])
    prediction = oracle.design(check, 5) @ lut
    expected = check @ matrix + [.1, .12, .15]
    expected[:, 0] += .05 * check[:, 0] * check[:, 1]
    assert all(s["success"] for s in states), states
    assert np.max(np.abs(prediction - expected)) < 1e-4


def test_second_derivative_scale_and_complementary_blocks():
    for d in (5, 9):
        grid = np.stack(np.meshgrid(*([np.linspace(0, 1, d)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
        derivatives = oracle.curvature(d) @ (grid[:, 0] ** 2) * (d - 1) ** 2
        n = (d - 2) * d**2
        np.testing.assert_allclose(derivatives[:n], 2, atol=1e-12)
        np.testing.assert_allclose(derivatives[n:], 0, atol=1e-12)
    c = {"block_size": 64, "block_border_exclusion": 4, "seed": 48, "max_fit_pixels": 30000}
    a, b = oracle.block_split(128, 192, c, 2, 0)
    cfit, dcheck = oracle.block_split(128, 192, c, 2, 1)
    np.testing.assert_array_equal(a, dcheck)
    np.testing.assert_array_equal(b, cfit)
    assert not np.intersect1d(a, b).size


def test_known_curved_lattice_with_nonzero_regularizer():
    rng = np.random.default_rng(49)
    d = 5
    grid = np.stack(np.meshgrid(*([np.linspace(0, 1, d)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    known = .15 + .55 * grid**2 + .1 * grid[:, [1, 2, 0]]
    assert np.max(np.abs(oracle.curvature(d) @ known)) > .05
    x, held_colors = rng.random((6000, 3)), rng.random((600, 3))
    target = oracle.design(x, d) @ known
    lut, states = oracle.fit_lut(x, target, d, 1e-6)
    assert all(s["success"] for s in states), states
    error = oracle.design(held_colors, d) @ (lut - known)
    assert np.max(np.abs(error)) < .002
    assert np.max(np.abs(oracle.curvature(d) @ lut)) > .04
