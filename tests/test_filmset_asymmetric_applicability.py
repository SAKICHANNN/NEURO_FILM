from __future__ import annotations

import numpy as np

from src.roll2film.cube_diffeomorphic_flow import (
    CubeDiffeomorphicColourFlow,
)
from src.roll2film.filmset_asymmetric_applicability import (
    all_gates_pass,
    applicability_gate_results,
    applicability_metrics,
    fit_sealed_policy,
    fit_standardized_pca,
    leave_one_case_oracle,
    leave_one_case_selection,
    operator_signature,
    oracle_metrics,
    selected_policy_loss,
)


def test_operator_signature_is_shared_centered_and_fixed_size() -> None:
    shared = CubeDiffeomorphicColourFlow.identity(
        axis_size=4, integration_steps=2
    )
    grid = np.zeros((4, 4, 4, 3), dtype=np.float64)
    grid[..., 0] = 0.01
    operator = CubeDiffeomorphicColourFlow(grid, integration_steps=2)
    signature = operator_signature(operator, shared)
    assert signature.shape == (200,)
    assert np.allclose(signature[:192].reshape(4, 4, 4, 3), grid)
    assert np.allclose(signature[192:195], [0.01, 0.0, 0.0])
    assert np.allclose(signature[195:198], [0.01, 0.0, 0.0])
    assert np.isclose(signature[199], 0.01)


def test_standardized_pca_is_repeat_exact_with_stable_signs() -> None:
    values = np.array(
        [
            [-2.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [2.0, -1.0, 0.0],
        ]
    )
    first = fit_standardized_pca(values, rank=2)
    second = fit_standardized_pca(values, rank=2)
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.scale, second.scale)
    assert np.array_equal(first.components, second.components)
    for component in first.components:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] >= 0.0


def test_leave_one_case_selection_excludes_same_case_operator() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    query = np.column_stack([np.cos(angles), np.sin(angles)])
    operator = query.copy()
    loss = 1.0 - 0.3 * (query @ operator.T)
    shared = np.full(8, 1.05)
    sources = np.arange(8, dtype=np.int64)
    selection = leave_one_case_selection(
        query,
        operator,
        loss,
        shared,
        sources,
        rank=2,
        ridge_penalty=0.1,
        ood_quantile=1.0,
    )
    assert np.all(selection.selected_indices >= 0)
    assert np.all(selection.selected_indices != sources)
    assert np.mean(selection.selected_loss) < np.mean(shared)
    oracle = leave_one_case_oracle(loss, sources)
    assert np.all(selection.selected_loss >= oracle - 1e-12)


def test_sealed_policy_falls_back_for_far_ood_query() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    query = np.column_stack([np.cos(angles), np.sin(angles)])
    operator = query.copy()
    loss = 1.0 - 0.3 * (query @ operator.T)
    policy = fit_sealed_policy(
        query,
        operator,
        loss,
        np.arange(8, dtype=np.int64),
        rank=2,
        ridge_penalty=0.1,
        ood_threshold=0.5,
    )
    indices, _margins, support = policy.select(
        np.array([[100.0, 100.0]])
    )
    assert indices.tolist() == [-1]
    assert support[0] > policy.ood_threshold
    policy_loss = selected_policy_loss(
        indices,
        np.full((1, 8), 0.1),
        np.array([0.75]),
    )
    assert policy_loss.tolist() == [0.75]


def test_applicability_gate_requires_every_control() -> None:
    shared = np.full(16, 1.0)
    oracle_loss = np.full(16, 0.7)
    policy = np.full(16, 0.8)
    medoid = np.full(16, 0.9)
    spatial = np.full(16, 0.95)
    random = np.full(16, 1.05)
    shuffled = np.full(16, 1.0)
    metrics = applicability_metrics(
        shared_loss=shared,
        oracle_loss=oracle_loss,
        policy_loss=policy,
        medoid_loss=medoid,
        spatial_loss=spatial,
        random_loss=random,
        shuffled_loss=shuffled,
        fallback_mask=np.zeros(16, dtype=bool),
        bootstrap_seed=7,
        bootstrap_samples=1000,
        bootstrap_confidence=0.95,
    )
    oracle = oracle_metrics(
        shared,
        oracle_loss,
        bootstrap_seed=8,
        bootstrap_samples=1000,
        bootstrap_confidence=0.95,
    )
    gates = {
        "minimum_eligible_case_count": 12,
        "minimum_oracle_improvement_over_shared_fraction": 0.1,
        "minimum_oracle_win_fraction": 0.75,
        "minimum_oracle_bootstrap_improvement_lower": 0.0,
        "minimum_policy_oracle_gap_closure": 0.25,
        "minimum_policy_improvement_over_shared_fraction": 0.02,
        "minimum_policy_improvement_over_medoid_fraction": 0.02,
        "minimum_policy_improvement_over_spatial_photometric_fraction": 0.02,
        "minimum_policy_improvement_over_random_fraction": 0.02,
        "minimum_policy_improvement_over_shuffled_fraction": 0.02,
        "minimum_policy_win_fraction_over_shared": 0.625,
        "minimum_policy_bootstrap_improvement_lower": 0.0,
        "maximum_ood_fallback_fraction": 0.25,
    }
    results = applicability_gate_results(
        metrics,
        oracle,
        eligible_case_count=24,
        structure_all=True,
        gates=gates,
    )
    assert all_gates_pass(results)
    results["policy_beats_medoid"] = False
    assert not all_gates_pass(results)
