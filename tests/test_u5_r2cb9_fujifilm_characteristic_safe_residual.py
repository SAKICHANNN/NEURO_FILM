from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_photographic import _compiled_curve
from src.eval.fujifilm_characteristic_safe_residual import (
    CONTRACT_SHA256,
    FujifilmCharacteristicSafeResidualError,
    apply_characteristic_safe_residual,
    load_contract,
)
from src.eval.fujifilm_dye_basis_measured_conformance import hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import _new_boundary_fraction

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb9_fujifilm_characteristic_safe_residual_v1.json"


def _curve():
    config = load_contract(CONFIG)
    parent = load_cb6(ROOT / config["parents"]["cb6_contract_path"])
    return _compiled_curve(parent)


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB9"


def test_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmCharacteristicSafeResidualError, match="hash drift"):
        load_contract(changed)


def test_analytical_scale_prevents_new_boundaries_and_preserves_source() -> None:
    epsilon = 1.0 / 510.0
    ramp = np.linspace(0.0, 1.0, 257 * 257, dtype=np.float32)
    source = np.stack(
        (ramp, np.roll(ramp, 7919), np.roll(ramp, 24011)), axis=-1
    ).reshape(257, 257, 3)
    before = source.copy()
    output, alpha = apply_characteristic_safe_residual(
        source,
        _curve(),
        nominal_strength=0.2,
        boundary_epsilon=epsilon,
    )
    assert np.array_equal(source, before)
    assert _new_boundary_fraction(source, output, epsilon) == 0.0
    assert np.min(output) >= 0.0 and np.max(output) <= 1.0
    assert np.min(alpha) >= 0.0 and np.max(alpha) <= 1.0
    assert np.count_nonzero(alpha < 1.0) > 0


def test_safe_scale_is_shared_across_rgb_residual() -> None:
    source = np.array([[[0.003, 0.2, 0.7], [0.4, 0.5, 0.6]]], dtype=np.float32)
    curve = _curve()
    output, alpha = apply_characteristic_safe_residual(
        source,
        curve,
        nominal_strength=0.2,
        boundary_epsilon=1.0 / 510.0,
    )
    raw = np.asarray(curve(source.astype(np.float64)), dtype=np.float64) - source
    applied = output.astype(np.float64) - source
    for pixel in range(source.shape[1]):
        material = np.abs(raw[0, pixel]) > 1e-8
        ratios = applied[0, pixel, material] / (0.2 * raw[0, pixel, material])
        assert np.allclose(ratios, float(alpha[0, pixel]), atol=2e-5, rtol=0.0)


def test_invalid_source_fails_before_execution() -> None:
    with pytest.raises(FujifilmCharacteristicSafeResidualError, match="invalid"):
        apply_characteristic_safe_residual(
            np.array([[[-0.1, 0.0, 0.0]]], dtype=np.float32),
            _curve(),
            nominal_strength=0.2,
            boundary_epsilon=1.0 / 510.0,
        )
