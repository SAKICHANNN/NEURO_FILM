from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.real_uniform_grain_nps import standardized_quadratic_residual
from src.eval.real_uniform_higher_order_structure import (
    RealUniformHigherOrderError,
    compare_feature_sets,
    evaluate_real_uniform_higher_order_structure,
    structure_features,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cf_real_uniform_higher_order_structure_v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_standardized_residual_is_zero_mean_unit_rms() -> None:
    y, x = np.mgrid[:64, :64]
    values = 100.0 + 0.1 * x + 0.2 * y + np.sin(x * 0.7) + np.cos(y * 0.4)
    residual = standardized_quadratic_residual(values)
    assert float(np.mean(residual)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.sqrt(np.mean(np.square(residual)))) == pytest.approx(1.0)


def test_structure_features_are_finite_and_topology_sensitive() -> None:
    contract = _contract()
    rng = np.random.default_rng(2608026201)
    gaussian = rng.normal(size=(512, 512))
    bounded = np.clip(gaussian, -2.0, 2.0)
    first = structure_features(gaussian, contract, relative_to_mean=False)
    second = structure_features(bounded, contract, relative_to_mean=False)
    assert set(first) == {
        "marginal_quantiles",
        "local_rms_quantiles",
        "excursion_topology",
    }
    assert all(np.all(np.isfinite(value)) for value in first.values())
    assert not np.array_equal(
        first["marginal_quantiles"], second["marginal_quantiles"]
    )
    compared = compare_feature_sets(second, first, second)
    assert compared["bounded_scan_win"] is True
    assert compared["median_metric_ratio"] == pytest.approx(0.0)


def test_contract_rejects_gate_drift() -> None:
    contract = _contract()
    validate_contract(contract)
    contract["automatic_gates"]["minimum_confirmation_scans_won"] = 1
    with pytest.raises(RealUniformHigherOrderError):
        validate_contract(contract)


@pytest.mark.skipif(
    not (ROOT / "data/external/wikimedia_uniform_grain_v1").is_dir()
    or not (ROOT / "data/external/wikimedia_bw_uniform_grain_v1").is_dir(),
    reason="exact local uniform-scan cohorts are unavailable",
)
def test_exact_real_uniform_higher_order_report_is_repeatable() -> None:
    contract = _contract()
    first = evaluate_real_uniform_higher_order_structure(contract, ROOT)
    second = evaluate_real_uniform_higher_order_structure(contract, ROOT)
    assert first == second
    assert first["development_summary"]["source_count"] == 8
    assert first["confirmation_summary"]["source_count"] == 3
