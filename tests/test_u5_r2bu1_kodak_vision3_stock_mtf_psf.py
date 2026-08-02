from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.kodak_vision3_stock_mtf_psf import (
    StockMtfPsfError,
    evaluate_stock_psf,
    fit_sigma,
    gaussian_mtf,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2bu1_kodak_vision3_stock_mtf_psf_v1.json"
TRACE = ROOT / "configs/data/kodak_vision3_mtf_curve_pixels_v1.json"


def test_contract_rejects_relaxed_shared_gate(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["minimum_overall_improvement_over_shared_fraction"] = 0.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StockMtfPsfError, match="contract drift"):
        load_contract(path)


def test_gaussian_sigma_grid_recovers_exact_synthetic_scale() -> None:
    frequencies = np.asarray([25.0, 35.0, 45.0, 60.0], dtype=np.float64)
    grid = 0.25 + 0.025 * np.arange(991, dtype=np.float64)
    target = gaussian_mtf(frequencies, 7.5)
    sigma, loss = fit_sigma(frequencies, target, grid)
    assert sigma == 7.5
    assert loss == pytest.approx(0.0, abs=1e-30)


@pytest.mark.skipif(not TRACE.is_file(), reason="exact Kodak MTF traces unavailable")
def test_exact_stock_psf_evaluation_is_repeatable() -> None:
    contract = load_contract(CONTRACT)
    first_report, first_bundle = evaluate_stock_psf(contract, ROOT)
    second_report, second_bundle = evaluate_stock_psf(contract, ROOT)
    assert first_report == second_report
    assert first_bundle == second_bundle
    assert len(first_report["confirmation_rows"]) == 9
    assert first_report["decision"] in {
        "retain_isolated_stock_specific_positive_psf_source_prior",
        "close_one_gaussian_stock_mtf_bank_without_rescue",
    }
