from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.finite_window_nps_recovery import (
    FiniteWindowNPSRecoveryError,
    evaluate_finite_window_nps_recovery,
    finite_window_periodogram,
    load_contract,
    radial_bin_means,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bn_finite_window_nps_recovery_v1.json"


def test_p4bn_contract_is_frozen_before_recovery() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4bn_finite_window_nps_recovery_contract.v1"
    )
    assert payload["estimator"]["crop_sizes"] == [128, 256]
    assert payload["estimator"]["windows"] == ["rectangular", "hann"]
    assert payload["estimator"]["post_score_window_or_bin_tuning_allowed"] is False
    assert payload["evaluation"]["maximum_256_hann_worst_log10_rmse"] == 0.4


def test_finite_window_periodogram_preserves_white_noise_scale() -> None:
    field = np.arange(64, dtype=np.float64).reshape(8, 8)
    rectangular = finite_window_periodogram(field, 0.001, "rectangular")
    hann = finite_window_periodogram(field, 0.001, "hann")
    assert rectangular.shape == field.shape
    assert hann.shape == field.shape
    assert np.all(rectangular >= 0.0)
    assert np.all(hann >= 0.0)
    assert rectangular[0, 0] < 1e-24
    assert hann[0, 0] < 1e-24


def test_radial_bin_means_rejects_empty_bins() -> None:
    with pytest.raises(ValueError, match="no frequency samples"):
        radial_bin_means(np.ones((4, 4)), 1.0, np.asarray([0.1, 0.2]))


def test_p4bn_contract_rejects_window_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["estimator"]["windows"] = ["hann"]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FiniteWindowNPSRecoveryError, match="contract drift"):
        load_contract(path)


def test_finite_window_nps_recovery_formal_decision() -> None:
    report = evaluate_finite_window_nps_recovery(load_contract(CONTRACT), ROOT)
    assert report["profile_count"] == 5
    assert report["decision"] in {
        "retain_hann_finite_window_nps_measurement_bridge",
        "close_direct_finite_window_nps_recovery",
    }
    assert report["maximum_periodic_control_log10_rmse"] < 1e-10
    assert report["gate_results"]["repeat_exact"] is True

