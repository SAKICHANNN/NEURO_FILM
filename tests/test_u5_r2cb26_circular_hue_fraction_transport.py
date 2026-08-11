from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.circular_hue_fraction_transport import (
    CircularHueFractionTransportError,
    circular_hue_fraction_transport_target,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def test_cb26_target_is_in_gamut_and_preserves_base_luminance() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb26_circular_hue_fraction_transport_v1.json"
    )
    assert contract["experiment_id"] == "U5.R2CB26"
    rng = np.random.default_rng(2600)
    base = rng.uniform(0.01, 0.99, size=(71, 79, 3)).astype(np.float32)
    ao6 = np.clip(0.15 + 0.72 * base[..., [1, 2, 0]], 0.0, 1.0).astype(np.float32)
    target = circular_hue_fraction_transport_target(base, ao6, weights=WEIGHTS)
    assert np.isfinite(target).all()
    assert float(np.min(target)) >= -1e-7
    assert float(np.max(target)) <= 1.0 + 1e-7
    np.testing.assert_allclose(
        np.sum(target.astype(np.float64) * WEIGHTS, axis=-1),
        np.sum(base.astype(np.float64) * WEIGHTS, axis=-1),
        atol=5e-8,
    )


def test_cb26_exact_neutral_pixels_remain_neutral() -> None:
    rng = np.random.default_rng(2601)
    base = rng.uniform(0.05, 0.95, size=(37, 41, 3)).astype(np.float32)
    ao6 = np.clip(base[..., [2, 0, 1]] * 0.8 + 0.1, 0.0, 1.0).astype(np.float32)
    neutral_luma = np.linspace(0.2, 0.8, base.shape[1], dtype=np.float32)
    base[0] = neutral_luma[:, None]
    target = circular_hue_fraction_transport_target(base, ao6, weights=WEIGHTS)
    np.testing.assert_array_equal(target[0, :, 0], target[0, :, 1])
    np.testing.assert_array_equal(target[0, :, 1], target[0, :, 2])


def test_cb26_rejects_fraction_slope_outside_envelope() -> None:
    rng = np.random.default_rng(2602)
    base = rng.uniform(0.05, 0.95, size=(61, 67, 3)).astype(np.float32)
    ao6 = np.full_like(base, 0.5)
    ao6[..., 0] += 1e-3 * (base[..., 0] - 0.5)
    with pytest.raises(CircularHueFractionTransportError, match="slope envelope"):
        circular_hue_fraction_transport_target(base, ao6, weights=WEIGHTS)
