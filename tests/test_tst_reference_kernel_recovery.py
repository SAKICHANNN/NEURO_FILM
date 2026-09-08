import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.eval import tst_reference_kernel as original
from src.eval.tst_reference_kernel_recovery import RecoveryKernel, block_preconditioner


def problem():
    rng = np.random.default_rng(20260908)
    groups = []
    for indices in ([0, 1], [2]):
        x = rng.random((250, 3))
        targets = [x**(.8+.5*j) for j in range(len(indices))]
        groups.append(original.sufficient(x, targets, list(indices), 3))
    return rng, groups, original.regularizer(3)


@pytest.mark.parametrize("collapsed", [False, True])
def test_block_preconditioner_preserves_original_beta_hessian(collapsed):
    rng, groups, reg = problem()
    rank = 2 if collapsed else 3
    orthogonal, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
    basis = orthogonal@np.diag(np.sqrt([.004, 30.] if collapsed else [.004, .3, 30.]))
    phi = basis[[0, 0, 1]] if collapsed else basis
    beta = 2e-6/(3*27)
    operator, _, rhs, _ = original.normal_system(phi, groups, reg, beta)
    h0, _, _, _ = original.normal_system(phi, groups, reg, 0.)
    preconditioner, audit = block_preconditioner(phi, groups, reg)
    eye = np.eye(rank*27)
    dense = np.column_stack([operator@e for e in eye])
    inverse_test = np.column_stack([preconditioner@(h0@e) for e in eye])
    np.testing.assert_allclose(inverse_test, eye, atol=1e-10)
    np.testing.assert_allclose(dense-np.column_stack([h0@e for e in eye]), beta*eye, atol=1e-15)
    assert audit["kind"] == ("collapsed_source_blocks" if collapsed else "full_row_blocks")
    proxy = RecoveryKernel(np.empty((3, 27, 3)), [], rtol=1e-8)
    recovered_h, recovered_m, recovered_rhs, _ = proxy.normal_system(phi, groups, reg, beta)
    np.testing.assert_array_equal(recovered_rhs, rhs)
    np.testing.assert_array_equal(recovered_h@eye[1], operator@eye[1])
    values, state = proxy.solve_channel(recovered_h, recovered_m, rhs[:, :, 0])
    assert state["solved"] and not state["reused"] and state["initialization"] == "zeros"
    relative = np.linalg.norm(dense@values.ravel()-rhs[:, :, 0].ravel())/np.linalg.norm(rhs[:, :, 0])
    assert relative <= 1e-8
    expected = np.linalg.solve(dense, rhs[:, :, 0].ravel())
    np.testing.assert_allclose(values.ravel(), expected, rtol=1e-4, atol=1e-6)
    objective = lambda v: v@dense@v-2*v@rhs[:, :, 0].ravel()
    np.testing.assert_allclose(objective(values.ravel()), objective(expected), atol=1e-12)


def test_freegrid_reuses_only_verified_channels(monkeypatch):
    _, groups, reg = problem()
    operator, preconditioner, rhs, _ = original.normal_system(np.eye(3), groups, reg, 0.)
    solved = [original.solve_channel(operator, preconditioner, rhs[:, :, c]) for c in range(3)]
    values = np.stack([v for v, _ in solved], axis=-1)
    proxy = RecoveryKernel(values, [s for _, s in solved])
    h, m, b, _ = proxy.normal_system(np.eye(3), groups, reg, 0.)
    def forbidden(*args, **kwargs):
        raise AssertionError("freegrid must not invoke CG")
    monkeypatch.setattr(original, "solve_channel", forbidden)
    for c in range(3):
        reused, state = proxy.solve_channel(h, m, b[:, :, c])
        np.testing.assert_array_equal(reused, values[:, :, c])
        assert state["reused"] and state["iterations"] == 0
    with pytest.raises(AssertionError):
        proxy.solve_channel(h, m, b[:, :, 0])


def test_rank_preconditions_stop_without_model_fallback():
    _, groups, reg = problem()
    with pytest.raises(AssertionError):
        block_preconditioner(np.ones((3, 1)), groups, reg)
    with pytest.raises(AssertionError):
        block_preconditioner(np.array([[1., 0.], [0., 1.], [1., 1.]]), groups, reg)


def test_recovery_config_preserves_science_and_subtracts_original_budget():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("recovery_entry", root/"scripts/run_tst_reference_kernel_recovery.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    runtime = entry.runtime()
    cfg, state, _ = runtime.preflight()
    assert cfg["cpu_seconds"]+state["recovery"]["original_cpu_seconds"] == 3600
    assert abs(cfg["wall_seconds_per_stage"]+state["recovery"]["original_wall_seconds"]-3600) < 1e-9
    assert state["recovery"]["freegrid_reused_channels"] == 3
    assert cfg["maxiter"] == 1500 and cfg["rtol"] == 1e-8
