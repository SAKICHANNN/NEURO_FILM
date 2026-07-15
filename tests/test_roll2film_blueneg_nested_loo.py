from __future__ import annotations

import numpy as np

from src.roll2film.blueneg_nested_loo import (
    count_preserving_label_permutation,
    nearest_other_rolls,
    robust_standardize,
    source_descriptor,
    stable_select,
    style_match_output,
)


def test_stable_select_is_deterministic_and_excludes_query() -> None:
    values = ["a", "b", "c", "d", "e"]
    first = stable_select(values, count=3, seed=7, namespace="fold", exclude="c")
    second = stable_select(values, count=3, seed=7, namespace="fold", exclude="c")
    assert first == second
    assert len(first) == 3
    assert "c" not in first


def test_label_permutation_preserves_counts() -> None:
    labels = {f"f{index}": "a" if index < 4 else "b" for index in range(10)}
    shuffled = count_preserving_label_permutation(labels, seed=19)
    assert sorted(shuffled.values()) == sorted(labels.values())
    assert shuffled == count_preserving_label_permutation(labels, seed=19)


def test_descriptor_retrieval_excludes_query_roll() -> None:
    descriptors = {
        "q": np.array([0.0, 0.0]),
        "same": np.array([0.01, 0.0]),
        "near": np.array([0.1, 0.0]),
        "far": np.array([2.0, 0.0]),
    }
    rolls = {"q": "r1", "same": "r1", "near": "r2", "far": "r3"}
    assert nearest_other_rolls("q", descriptors, rolls, count=2) == ("near", "far")


def test_descriptor_and_robust_scaling_are_finite() -> None:
    rng = np.random.default_rng(3)
    descriptors = {
        "a": source_descriptor(rng.uniform(0.01, 0.8, size=(64, 3))),
        "b": source_descriptor(rng.uniform(0.02, 0.9, size=(64, 3))),
    }
    standardized = robust_standardize(descriptors)
    assert standardized["a"].shape == (11,)
    assert np.all(np.isfinite(np.stack(list(standardized.values()))))


def test_style_match_uses_source_and_reaches_reference_strength() -> None:
    rng = np.random.default_rng(11)
    source = rng.uniform(0.05, 0.6, size=(256, 3))
    candidate = np.clip(source * 1.25 + 0.02, 0.0, 1.0)
    reference = source + 0.5 * (candidate - source)
    matched, alpha, achieved = style_match_output(source, candidate, reference)
    assert matched.shape == source.shape
    assert 0.45 < alpha < 0.55
    assert achieved > 0.0
