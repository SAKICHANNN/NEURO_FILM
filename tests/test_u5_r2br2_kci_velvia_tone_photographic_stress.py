from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.eval.kci_velvia_tone_photographic_stress import (
    _gradient_inversion_fraction,
    apply_fixed_tone,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2br2_kci_velvia_tone_photographic_stress_v1.json"


def test_contract_is_fixed_photographic_stress_only() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["population"]["source_count_exact"] == 12
    assert config["operator"]["fit_on_photographs_forbidden"]
    assert config["operator"]["hard_clipping_forbidden"]
    assert not config["production_integration_allowed"]
    assert not config["stock_response_claim_allowed"]


def test_fixed_tone_is_bounded_nonmutating_and_exposes_local_order_risk() -> None:
    axis = np.linspace(0.05, 0.85, 32, dtype=np.float32)
    source = np.stack(np.meshgrid(axis, axis[:24]), axis=-1)
    source = np.concatenate(
        (source, np.full(source.shape[:2] + (1,), 0.2, dtype=np.float32)), axis=-1
    )
    before = source.copy()
    curve = PchipInterpolator(
        np.asarray([0.0, 20.0, 50.0, 80.0, 100.0]),
        np.asarray([0.0, 30.0, 62.0, 90.0, 100.0]),
        extrapolate=False,
    )
    output, source_lab, output_lab, scale = apply_fixed_tone(source, curve)
    assert np.array_equal(source, before)
    assert np.isfinite(output).all()
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.min(scale) >= 0.0
    assert np.max(scale) <= 1.0
    # The frozen PCHIP is monotone, but colour-dependent per-pixel gamut
    # compression can change local L* ordering.  BR2 measures and gates that
    # mechanism rather than hiding it in a unit fixture.
    assert np.all(np.diff(curve(np.linspace(0.0, 100.0, 1001))) >= 0.0)
    assert _gradient_inversion_fraction(
        source_lab[..., 0], output_lab[..., 0], epsilon=0.0001
    ) > 0.0


def test_gradient_inversion_detector_rejects_reversal() -> None:
    source = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    output = np.asarray([[1.0, 3.0, 2.0]], dtype=np.float32)
    assert _gradient_inversion_fraction(source, output, epsilon=0.0001) > 0.0
