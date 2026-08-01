from __future__ import annotations

import numpy as np

from src.eval.fivek_pairwise_compatibility import (
    fit_projection,
    fit_ridge_ranker,
    pair_features,
    predict_ranker,
    project_features,
    rank_queries,
    training_matrix,
)


def test_projection_is_repeat_exact_and_sign_stable() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(20, 12))
    first = fit_projection(features, 6)
    second = fit_projection(features, 6)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
    projected = project_features(features, *first)
    assert projected.shape == (20, 6)
    for component in first[2]:
        assert component[np.argmax(np.abs(component))] >= 0.0


def test_pair_features_preserve_ordered_query_case_roles() -> None:
    query = np.asarray([[1.0, 2.0]])
    case = np.asarray([[3.0, 5.0]])
    assert np.array_equal(
        pair_features(query, case),
        np.asarray([[1.0, 2.0, 3.0, 5.0, 2.0, 3.0, 3.0, 10.0]]),
    )


def test_ridge_ranker_recovers_synthetic_linear_pair_scores() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(200, 12))
    coefficient = rng.normal(size=12)
    y = x @ coefficient + 0.25
    model = fit_ridge_ranker(x, y, 1.0e-6)
    predicted = predict_ranker(model, x)
    assert np.all(np.isfinite(predicted))
    assert float(np.max(np.abs(predicted - y))) < 1.0e-6


def test_rank_queries_returns_one_hard_case_per_query() -> None:
    projected = np.asarray(
        [[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64
    )
    query = np.repeat(projected[2][None, :], 3, axis=0)
    x = pair_features(query, projected[np.asarray([0, 1, 3])])
    y = np.asarray([2.0, 1.0, 3.0])
    model = fit_ridge_ranker(x, y, 1.0e-6)
    chosen = rank_queries(
        projected=projected,
        query_indices=np.asarray([2]),
        case_indices=np.asarray([0, 1, 3]),
        model=model,
    )
    assert chosen.shape == (1,)
    assert int(chosen[0]) == 1


def test_shuffled_training_targets_are_deterministic_and_distinct() -> None:
    projected = np.arange(18, dtype=np.float64).reshape(6, 3)
    errors = np.abs(np.subtract.outer(np.arange(6), np.arange(6))).astype(float)
    indices = np.arange(6)
    _, original = training_matrix(projected, indices, errors)
    _, first = training_matrix(projected, indices, errors, shuffled_seed=19)
    _, second = training_matrix(projected, indices, errors, shuffled_seed=19)
    assert np.array_equal(first, second)
    assert not np.array_equal(original, first)
