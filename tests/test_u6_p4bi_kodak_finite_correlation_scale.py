from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.kodak_finite_correlation_scale import (
    KodakFiniteCorrelationScaleError,
    evaluate_scale,
    load_contract,
)
from src.film_physics.gaussian_aperture_variance import (
    gaussian_covariance_aperture_rms_ratio,
    gaussian_covariance_disk_average_variance,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bi_kodak_finite_correlation_scale_v1.json"


def test_contract_rejects_density_specific_fit(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate"]["density_specific_parameters_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KodakFiniteCorrelationScaleError, match="contract drift"):
        load_contract(path)


def test_gaussian_disk_variance_limits_and_selwyn_limit() -> None:
    apertures = np.asarray([7.25, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0])
    variance = gaussian_covariance_disk_average_variance(apertures, 4.0)
    assert np.all(np.diff(variance) < 0.0)
    ratios = gaussian_covariance_aperture_rms_ratio(
        apertures,
        reference_aperture_micrometres=48.0,
        correlation_scale_micrometres=0.1,
    )
    selwyn = 48.0 / apertures
    assert np.max(np.abs(ratios / selwyn - 1.0)) < 0.01
    assert ratios[3] == pytest.approx(1.0, abs=1e-15)


def test_frozen_scale_evaluator_repeats() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_scale(contract, ROOT)
    second = evaluate_scale(contract, ROOT)
    assert first == second
    assert first["development_group_count"] == 12
    assert first["confirmation_group_count"] == 11
    assert first["confirmation_comparison_count"] == 66
