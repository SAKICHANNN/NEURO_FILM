from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.vfgs_frequency_shaping_approximation import (
    VfgsFrequencyShapingApproximationError,
    evaluate_vfgs_frequency_shaping_approximation,
    validate_contract,
    vfgs_frequency_window,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8br_vfgs_frequency_shaping_approximation_v1.json"


def test_p8br_contract_freezes_current_primary_sources_and_no_ml_runtime() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"].endswith("contract.v1")
    assert contract["primary_sources"]["vfgs"]["commit"] == (
        "fbf4bd95058e934fb7246edd5e7fb8d6c9ed0ec0"
    )
    assert contract["primary_sources"]["vfgs"]["core_sha256"] == (
        "8051fbefc6fef8395e678a4a105457fc799a16044e51a6123f53fd0a28e0946a"
    )
    assert contract["primary_sources"]["fga_nn"]["pdf_sha256"] == (
        "cf55d2ab2cadd7d80cd50dde4c1fc2db286b7e6b203fe1de7cc0f46749c9e30a"
    )
    assert contract["model"]["cutoff_candidates_inclusive"] == [2, 14]
    assert contract["model"]["soft_window_k"] == 1.118
    assert contract["model"]["learned_predictor_used"] is False
    assert contract["roles"]["refit_rescale_or_cutoff_change_on_confirmation"] is False


def test_rectangular_window_matches_exact_source_cutoff_semantics() -> None:
    window = vfgs_frequency_window(cutoff=8, softness_k=0.0)
    assert window.shape == (64,)
    assert np.array_equal(window[:36], np.ones(36))
    assert np.array_equal(window[36:], np.zeros(28))


def test_default_soft_window_detects_source_table_exhaustion() -> None:
    with pytest.raises(VfgsFrequencyShapingApproximationError, match="exhausts"):
        vfgs_frequency_window(cutoff=9, softness_k=1.118)
    for cutoff in (*range(2, 9), *range(10, 15)):
        window = vfgs_frequency_window(cutoff=cutoff, softness_k=1.118)
        assert window.shape == (64,)
        assert np.all(np.isfinite(window))


def test_contract_rejects_candidate_range_drift() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    validate_contract(contract)
    contract["model"]["cutoff_candidates_inclusive"] = [2, 13]
    with pytest.raises(VfgsFrequencyShapingApproximationError):
        validate_contract(contract)


def test_source_conformance_closes_before_scan_read() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = evaluate_vfgs_frequency_shaping_approximation(contract, ROOT)
    second = evaluate_vfgs_frequency_shaping_approximation(contract, ROOT)
    assert first == second
    assert first["automatic_pass"] is False
    assert first["decision"] == "close_default_vfgs_soft_window_before_scan_read"
    assert first["source_conformance"]["invalid_soft_cutoffs"] == [9]
    assert first["development_source_reads"] == 0
    assert first["confirmation_source_reads"] == 0
