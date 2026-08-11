from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.monotone_fraction_quantile_transport import (
    _empirical_quantile_transport,
    load_contract,
    monotone_fraction_quantile_transport_target,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def test_cb27_empirical_transport_is_monotone_and_bounded() -> None:
    base = np.asarray([[0.8, 0.2, 0.6, 0.4]], dtype=np.float64)
    target = np.asarray([[0.1, 0.3, 0.7, 0.9]], dtype=np.float64)
    valid = np.ones_like(base, dtype=bool)
    mapped = _empirical_quantile_transport(
        base, target, valid, valid, minimum_valid_fraction=1e-4
    )
    order = np.argsort(base.reshape(-1), kind="stable")
    assert np.all(np.diff(mapped.reshape(-1)[order]) >= 0.0)
    assert float(np.min(mapped)) >= 0.0
    assert float(np.max(mapped)) <= 1.0


def test_cb27_target_is_in_gamut_and_preserves_base_luminance() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb27_monotone_fraction_quantile_transport_v1.json"
    )
    assert contract["experiment_id"] == "U5.R2CB27"
    rng = np.random.default_rng(2700)
    base = rng.uniform(0.01, 0.99, size=(71, 79, 3)).astype(np.float32)
    ao6 = np.clip(0.1 + 0.8 * base[..., [1, 2, 0]], 0.0, 1.0).astype(np.float32)
    target = monotone_fraction_quantile_transport_target(base, ao6, weights=WEIGHTS)
    assert np.isfinite(target).all()
    assert float(np.min(target)) >= -1e-7
    assert float(np.max(target)) <= 1.0 + 1e-7
    np.testing.assert_allclose(
        np.sum(target.astype(np.float64) * WEIGHTS, axis=-1),
        np.sum(base.astype(np.float64) * WEIGHTS, axis=-1),
        atol=5e-8,
    )


def test_cb27_exact_neutral_pixels_remain_neutral() -> None:
    rng = np.random.default_rng(2701)
    base = rng.uniform(0.05, 0.95, size=(37, 41, 3)).astype(np.float32)
    ao6 = np.clip(base[..., [2, 0, 1]] * 0.8 + 0.1, 0.0, 1.0).astype(np.float32)
    neutral_luma = np.linspace(0.2, 0.8, base.shape[1], dtype=np.float32)
    base[0] = neutral_luma[:, None]
    target = monotone_fraction_quantile_transport_target(base, ao6, weights=WEIGHTS)
    np.testing.assert_array_equal(target[0, :, 0], target[0, :, 1])
    np.testing.assert_array_equal(target[0, :, 1], target[0, :, 2])
