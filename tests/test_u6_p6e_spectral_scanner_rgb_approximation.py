from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_spectral_scanner_rgb_approximation import (
    _predict_quadratic,
    evaluate_spectral_scanner_rgb_approximation,
    fit_nonnegative_row_sum_bounded_matrix,
    load_contract,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs" / "u6_p6e_spectral_scanner_rgb_approximation_v1.json"
)


def test_report_has_disjoint_frozen_splits_and_bounded_candidates() -> None:
    report = evaluate_spectral_scanner_rgb_approximation(
        ROOT, load_contract(CONTRACT)
    )
    assert report["dataset"]["row_counts"] == {
        "development": 1024,
        "confirmation": 512,
        "stress": 512,
    }
    assert report["dataset"]["group_counts"] == {
        "development": 8,
        "confirmation": 4,
        "stress": 4,
    }
    assert report["dataset"]["group_overlap"] == 0
    assert report["dataset"]["spectrum_hash_overlap"] == 0
    assert report["dataset"]["scanner_a_rgb_hash_overlap"] == 0
    for split in report["output_extrema"].values():
        for name, interval in split.items():
            assert 0.0 <= interval[0] <= interval[1] <= 1.0, name


def test_bounded_matrix_is_nonnegative_and_row_sum_bounded() -> None:
    rng = np.random.default_rng(20260729)
    source = rng.random((128, 3))
    target = rng.random((128, 3))
    matrix = fit_nonnegative_row_sum_bounded_matrix(source, target)
    assert np.all(matrix >= 0.0)
    assert np.all(np.sum(matrix, axis=1) <= 1.0 + 1e-15)
    prediction = source @ matrix.T
    assert np.min(prediction) >= 0.0
    assert np.max(prediction) <= 1.0


def test_quadratic_prediction_is_bounded_without_output_clipping() -> None:
    rng = np.random.default_rng(20260730)
    source = rng.random((64, 3))
    coefficients = rng.normal(size=(10, 3)) * 20.0
    prediction = _predict_quadratic(source, coefficients)
    assert np.all(np.isfinite(prediction))
    assert np.min(prediction) >= 0.0
    assert np.max(prediction) <= 1.0


def test_report_repeats_and_reproduces_p6d_lower_bound(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_spectral_scanner_rgb_approximation(ROOT, contract)
    second = evaluate_spectral_scanner_rgb_approximation(ROOT, contract)
    assert first == second
    assert first["checks"]["p6d_metamer_lower_bound"] is True
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    assert write_report(first, path_a) == write_report(second, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_duplicate_group_seed_fails_disjoint_gate() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["data_contract"]["confirmation"]["group_seeds"][0] = contract[
        "data_contract"
    ]["development"]["group_seeds"][0]
    report = evaluate_spectral_scanner_rgb_approximation(ROOT, contract)
    assert report["checks"]["group_disjoint"] is False
    assert report["automatic_pass"] is False


def test_parent_report_hash_drift_fails_closed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["parents"]["p6d_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="P6D report hash mismatch"):
        evaluate_spectral_scanner_rgb_approximation(ROOT, contract)
