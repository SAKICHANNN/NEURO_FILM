import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.eval.tst_reference_response import (
    fit_projection,
    lattice_design,
    matched_heads,
    paired_rows,
    project,
    render,
    scoring_folds,
    teacher_factor,
    teacher_solve,
)


def test_independent_interpolation_identity_and_teacher():
    rng = np.random.default_rng(20260908)
    x = np.vstack([rng.random((4096, 3)), np.eye(3), np.zeros((1, 3)), np.ones((1, 3))])
    axis = np.linspace(0, 1, 7)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    residual = .025 * grid * (1-grid)
    independent = RegularGridInterpolator((axis, axis, axis), residual)(x)
    np.testing.assert_allclose(lattice_design(x) @ residual.reshape(-1, 3), independent, atol=1e-15)
    np.testing.assert_array_equal(render(x, np.zeros((343, 3))), x)
    target = x + independent
    factor = teacher_factor(x)
    fitted, error = teacher_solve(factor, target)
    assert error < 1e-8
    assert np.mean(np.abs(render(x, fitted)-target)) < .0002
    a, b = scoring_folds(128, 192, "source")
    assert len(a) <= 8192 and len(b) <= 8192 and not np.intersect1d(a, b).size


def test_fit_only_projection_constant_and_rank_padding():
    values = np.tile(np.arange(10)[:, None], (1, 8)).astype(float)
    values[:, 0] = 5
    model = fit_projection([values])
    projected = project(model, [values])
    assert projected.shape == (10, 6)
    assert np.all(projected[:, 1:] == 0)
    assert np.max(np.abs(project(model, [values+1000]))) <= 3


def test_matched_heads_capacity_and_source_only_switch():
    rng = np.random.default_rng(48)
    source = np.repeat(rng.normal(size=(20, 6)), 2, axis=0)
    reference = rng.normal(size=(40, 6))
    simple = rng.normal(size=(40, 6))
    ids = np.repeat(np.arange(20), 2)
    features = {"A": np.column_stack([np.ones(40), source, np.zeros_like(reference)]),
                "B": np.column_stack([np.ones(40), source, simple]),
                "C": np.column_stack([np.ones(40), source, reference])}
    labels = source @ rng.normal(size=(6, 12)) + reference @ rng.normal(size=(6, 12))
    heads = matched_heads(features, labels, ids)
    for head in heads.values():
        assert abs(head["achieved_df"]-4) < 1e-8
    a = features["A"] @ heads["A"]["coefficients"]
    np.testing.assert_allclose(a[::2], a[1::2], atol=0)
    c = features["C"] @ heads["C"]["coefficients"]
    swap = np.arange(40).reshape(-1, 2)[:, ::-1].ravel()
    np.testing.assert_allclose(features["C"][swap] @ heads["C"]["coefficients"], c[swap])
    pair_x, pair_y, weights = paired_rows(features["C"], labels, ids)
    assert np.all(pair_x[1::2, 0] == 0) and np.all(weights[1::2] == 2)
    np.testing.assert_allclose(pair_y[1::2], (labels[1::2]-labels[::2])/2)


def test_zero_rank_is_explicit_mean_only():
    features = {a: np.column_stack([np.ones(6), np.zeros((6, 12))]) for a in "ABC"}
    heads = matched_heads(features, np.arange(18).reshape(6, 3), [0, 0, 1, 1, 2, 2])
    for h in heads.values():
        assert h["target_df"] == 0 and np.all(h["coefficients"][1:] == 0)


def test_actual_entry_reference_only_swap_without_targets():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("response_entry", Path(__file__).resolve().parents[1]/"scripts/run_tst_reference_response.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    rng = np.random.default_rng(52)
    keys = [f"{g:02d}_{role}" for g in range(24) for role in ("X", "R1", "R2")]
    stats, vgg = {k: rng.random(30) for k in keys}, {k: rng.random(16) for k in keys}
    rows = [{"group": g, "target": j, "conditional": True, "values": rng.normal(size=(343, 3))}
            for g in range(12) for j in (1, 2)]
    model = entry.student_fit(stats, vgg, rows)
    old1 = entry.predicted_operators(model, stats, vgg, 13, 1)
    old2 = entry.predicted_operators(model, stats, vgg, 13, 2)
    for values in (stats, vgg):
        values["13_R1"], values["13_R2"] = values["13_R2"], values["13_R1"]
    new1 = entry.predicted_operators(model, stats, vgg, 13, 1)
    new2 = entry.predicted_operators(model, stats, vgg, 13, 2)
    for arm in "ABC":
        np.testing.assert_array_equal(new1[arm], old2[arm])
        np.testing.assert_array_equal(new2[arm], old1[arm])
    np.testing.assert_array_equal(old1["A"], old2["A"])
