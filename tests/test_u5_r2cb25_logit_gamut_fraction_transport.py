from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.logit_gamut_fraction_transport import (
    LogitGamutFractionTransportError,
    load_contract,
    logit_gamut_fraction_transport_target,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def test_cb25_target_is_in_gamut_and_preserves_base_luminance() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb25_logit_gamut_fraction_transport_v1.json"
    )
    assert contract["experiment_id"] == "U5.R2CB25"
    rng = np.random.default_rng(2500)
    base = rng.uniform(0.001, 0.999, size=(71, 79, 3)).astype(np.float32)
    ao6 = rng.uniform(0.001, 0.999, size=(71, 79, 3)).astype(np.float32)
    target = logit_gamut_fraction_transport_target(
        base,
        ao6,
        weights=WEIGHTS,
        minimum_transport_eigenvalue=0.01,
        maximum_transport_eigenvalue=10.0,
    )
    assert np.isfinite(target).all()
    assert float(np.min(target)) >= -1e-7
    assert float(np.max(target)) <= 1.0 + 1e-7
    np.testing.assert_allclose(
        np.sum(target.astype(np.float64) * WEIGHTS, axis=-1),
        np.sum(base.astype(np.float64) * WEIGHTS, axis=-1),
        atol=5e-8,
    )


def test_cb25_exact_neutral_target_remains_neutral() -> None:
    rng = np.random.default_rng(2501)
    luma = rng.uniform(0.1, 0.9, size=(37, 41, 1)).astype(np.float32)
    base = np.repeat(luma, 3, axis=-1)
    ao6 = np.repeat(np.clip(0.8 * luma + 0.1, 0.0, 1.0), 3, axis=-1)
    with pytest.raises(LogitGamutFractionTransportError, match="covariance envelope"):
        logit_gamut_fraction_transport_target(base, ao6, weights=WEIGHTS)


def test_cb25_rejects_non_float32_input() -> None:
    base = np.full((7, 9, 3), 0.5, dtype=np.float64)
    with pytest.raises(LogitGamutFractionTransportError, match="input drift"):
        logit_gamut_fraction_transport_target(base, base, weights=WEIGHTS)
