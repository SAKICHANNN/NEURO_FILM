from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ao8c_velvia_crossfit_case_selector import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.velvia_crossfit_case_selector import (
    VelviaCrossfitCaseError,
    case_knn_labels,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/u5_r2ao8c_velvia_crossfit_case_selector_v1.json"


def _config() -> dict:
    return load_config(CONFIG_PATH, expected_sha256=CONFIG_SHA256)


def test_contract_validates_exact_parent_and_outer_split() -> None:
    ao8b = validate_contract(ROOT, _config())
    assert ao8b["nested_evaluation"]["fold_count"] == 5


def test_contract_rejects_control_gate_drift() -> None:
    changed = copy.deepcopy(_config())
    changed["gates"][
        "minimum_selected_rgb_rmse_gain_over_best_domain_only_control"
    ] = 0.0
    with pytest.raises(VelviaCrossfitCaseError, match="contract"):
        validate_contract(ROOT, changed)


def test_config_hash_fails_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG_PATH, expected_sha256="0" * 64)


def test_case_knn_uniform_and_distance_votes_are_hard_sparse() -> None:
    cases = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.9, 0.9, 0.9],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([1, 1, 0, 0, 0], dtype=np.int8)
    query = np.asarray([[0.03, 0.0, 0.0], [0.95, 0.95, 0.95]])
    uniform = case_knn_labels(
        query, cases, labels, neighbors=3, vote="uniform"
    )
    distance = case_knn_labels(
        query, cases, labels, neighbors=3, vote="inverse_distance"
    )
    np.testing.assert_array_equal(uniform, np.asarray([1, 0]))
    np.testing.assert_array_equal(distance, np.asarray([1, 0]))


def test_case_knn_rejects_dense_even_k() -> None:
    with pytest.raises(VelviaCrossfitCaseError, match="invalid"):
        case_knn_labels(
            np.zeros((1, 3)),
            np.zeros((4, 3)),
            np.zeros(4, dtype=np.int8),
            neighbors=2,
            vote="uniform",
        )
