from __future__ import annotations

import numpy as np

from src.real_film.connected_stock_identifiability import (
    group_label_permutation_test,
    group_loo_centroid,
    paired_group_bootstrap_delta,
)


def _separable() -> tuple[np.ndarray, list[str], list[str]]:
    groups = [f"a{i}" for i in range(5) for _ in range(2)] + [f"b{i}" for i in range(5) for _ in range(2)]
    labels = ["a"] * 10 + ["b"] * 10
    values = [[i * 0.02] for i in range(5) for _ in range(2)] + [[10 + i * 0.02] for i in range(5) for _ in range(2)]
    return np.asarray(values), groups, labels


def test_group_loo_centroid_uses_equal_author_group_votes() -> None:
    features, groups, labels = _separable()
    result = group_loo_centroid(features, groups, labels)
    assert result["held_out_author_groups"] == 10
    assert result["author_label_units"] == 10
    assert result["balanced_accuracy"] == 1.0
    assert result["per_class_group_recall"] == {"a": 1.0, "b": 1.0}


def test_group_permutation_is_deterministic() -> None:
    features, groups, labels = _separable()
    first = group_label_permutation_test(features, groups, labels, permutations=31, seed=7)
    second = group_label_permutation_test(features, groups, labels, permutations=31, seed=7)
    assert first == second
    assert first["observed_balanced_accuracy"] == 1.0


def test_paired_bootstrap_reports_positive_primary_delta() -> None:
    features, groups, labels = _separable()
    primary = group_loo_centroid(features, groups, labels)
    control = group_loo_centroid(np.zeros_like(features), groups, labels)
    first = paired_group_bootstrap_delta(primary, control, iterations=200, seed=9)
    second = paired_group_bootstrap_delta(primary, control, iterations=200, seed=9)
    assert first == second
    assert first["observed_delta"] > 0
    assert first["ci95_low"] > 0


def test_multilabel_author_is_held_out_as_one_split_group() -> None:
    features, groups, labels = _separable()
    features = np.concatenate([features, np.asarray([[0.05], [10.05]])])
    groups = [*groups, "paired", "paired"]
    labels = [*labels, "a", "b"]
    result = group_loo_centroid(features, groups, labels)
    paired = [row for row in result["predictions"] if row["held_out_author_group"] == "paired"]
    assert result["held_out_author_groups"] == 11
    assert result["author_label_units"] == 12
    assert {row["true_label"] for row in paired} == {"a", "b"}
    assert all(row["correct"] for row in paired)
