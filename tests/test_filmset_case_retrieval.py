from __future__ import annotations

import numpy as np

from src.roll2film.filmset_case_retrieval import (
    CaseBankEvaluation,
    bootstrap_mean_improvement_ci,
    content_descriptor,
    decide_case_retrieval_branch,
    evaluate_wrong_recipe_oracle,
    oracle_gap_closure,
    standardized_nearest_indices,
)


def _samples() -> tuple[np.ndarray, np.ndarray]:
    cells = np.repeat(np.arange(4, dtype=np.int64), 4)
    first = np.linspace(0.05, 0.45, 16)[:, None] * np.array(
        [[1.0, 0.8, 0.6]]
    )
    second = np.linspace(0.55, 0.95, 16)[:, None] * np.array(
        [[0.6, 0.8, 1.0]]
    )
    return np.stack([first, second]), cells


def test_content_descriptors_use_input_samples_and_add_spatial_cells() -> None:
    samples, cells = _samples()
    global_only = content_descriptor(
        samples, cells, include_spatial_cells=False
    )
    spatial = content_descriptor(
        samples, cells, include_spatial_cells=True
    )
    assert global_only.shape == (2, 35)
    assert spatial.shape == (2, 55)
    assert np.all(np.isfinite(spatial))
    assert not np.allclose(global_only[0], global_only[1])


def test_standardized_nearest_indices_fit_scale_on_bank() -> None:
    bank = np.array([[0.0, 100.0], [1.0, 200.0], [2.0, 300.0]])
    query = np.array([[0.1, 110.0], [1.9, 290.0]])
    indices, margins = standardized_nearest_indices(bank, query)
    assert indices.tolist() == [0, 2]
    assert np.all(margins > 0.0)


def test_oracle_gap_closure_has_expected_endpoints() -> None:
    shared = np.array([0.10, 0.20])
    oracle = np.array([0.05, 0.10])
    assert oracle_gap_closure(shared, shared, oracle) == 0.0
    assert oracle_gap_closure(shared, oracle, oracle) == 1.0


def test_bootstrap_improvement_is_repeat_exact_and_positive() -> None:
    baseline = np.array([0.3, 0.4, 0.5, 0.6])
    challenger = baseline - 0.1
    first = bootstrap_mean_improvement_ci(
        baseline,
        challenger,
        seed=7,
        samples=1000,
        confidence=0.95,
    )
    second = bootstrap_mean_improvement_ci(
        baseline,
        challenger,
        seed=7,
        samples=1000,
        confidence=0.95,
    )
    assert first == second
    assert first["lower"] > 0.09


def test_case_branch_requires_oracle_before_retrieval() -> None:
    passing = {
        "structure_all": True,
        "minimum_eligible_bank": True,
        "oracle_improvement": True,
        "oracle_win_fraction": True,
        "oracle_bootstrap": True,
        "retrieval_gap_closure": True,
        "retrieval_beats_random": True,
        "retrieval_beats_shuffled": True,
    }
    assert (
        decide_case_retrieval_branch(passing)
        == "case_bank_oracle_and_retrieval_pass"
    )
    oracle_only = {**passing, "retrieval_gap_closure": False}
    assert decide_case_retrieval_branch(oracle_only) == "case_bank_oracle_only"
    no_oracle = {**passing, "oracle_improvement": False}
    assert (
        decide_case_retrieval_branch(no_oracle)
        == "case_bank_no_oracle_value"
    )


def test_wrong_recipe_control_reports_unavailable_for_invalid_bank() -> None:
    evaluation = CaseBankEvaluation(
        report={},
        eligible_operators=(),
        query_input_evaluation=np.zeros((2, 16, 3)),
        query_target_evaluation=np.zeros((2, 16, 3)),
        oracle_rmse=np.zeros(2),
    )
    result = evaluate_wrong_recipe_oracle(evaluation, ())
    assert result["status"] == (
        "unavailable_insufficient_wrong_recipe_bank"
    )
    assert result["wrong_operator_count"] == 0
    assert result["best_wrong_recipe_rmse"] is None
