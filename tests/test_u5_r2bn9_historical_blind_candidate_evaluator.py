from __future__ import annotations

import numpy as np
import pytest

from src.eval.historical_blind_candidate_evaluator import (
    HistoricalBlindCandidateEvaluatorError,
    _cross_validate,
    _experiment_global_baseline,
    _utility_features,
)


def test_utility_feature_schema_and_identity_delta() -> None:
    grid = np.linspace(0.05, 0.95, 24 * 32 * 3, dtype=np.float32).reshape(24, 32, 3)
    features, names, base = _utility_features(grid, grid.copy())
    assert features.shape == (88,)
    assert len(names) == 88
    assert base == 52
    delta_names = [index for index, name in enumerate(names) if name.startswith("delta_")]
    assert np.max(np.abs(features[delta_names])) == pytest.approx(0.0, abs=1e-12)


def test_utility_features_reject_shape_mismatch() -> None:
    with pytest.raises(HistoricalBlindCandidateEvaluatorError, match="shape mismatch"):
        _utility_features(
            np.zeros((8, 8, 3), dtype=np.float32),
            np.zeros((7, 8, 3), dtype=np.float32),
        )


def test_source_group_cross_validation_can_learn_interaction() -> None:
    units = []
    features = {}
    for source_index in range(12):
        source_id = f"s{source_index:02d}"
        winner = "a" if source_index % 2 == 0 else "b"
        units.append(
            {
                "experiment_id": "e",
                "source_id": source_id,
                "arms": ["a", "b"],
                "majority_arm": winner,
            }
        )
        sign = 1.0 if source_index % 2 == 0 else -1.0
        features[("e", source_id, "a")] = np.asarray([sign, 1.0])
        features[("e", source_id, "b")] = np.asarray([-sign, 1.0])
    result = _cross_validate(
        units=units,
        features=features,
        groups=[unit["source_id"] for unit in units],
        feature_slice=slice(0, 2),
        regularization_c=1.0,
        seed=7,
    )
    baseline = _experiment_global_baseline(
        units, [unit["source_id"] for unit in units]
    )
    assert result["accuracy"] == 1.0
    assert baseline["accuracy"] == 0.0
