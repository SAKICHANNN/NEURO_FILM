from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_characteristic_forward_proxy import (
    CONTRACT_SHA256,
    FujifilmCharacteristicForwardError,
    _apply_curve,
    _characteristic_curves,
    _fit_characteristic_layer,
    evaluate_characteristic_forward_proxy,
    load_contract,
)
from src.eval.fujifilm_dye_basis_measured_conformance import hash_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb6_fujifilm_characteristic_forward_proxy_v1.json"


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB6"


def test_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmCharacteristicForwardError, match="contract hash drift"):
        load_contract(changed)


def test_characteristic_trace_is_monotone_and_nonnegative() -> None:
    config = load_contract(CONFIG)
    exposure, curves, floors = _characteristic_curves(config)
    assert np.all(floors > 0.0)
    for curve in curves:
        values = curve(exposure)
        assert np.all(values >= 0.0)
        assert np.all(np.diff(values) <= 0.0)


def test_layer_fit_recovers_exact_synthetic_positive_exposure() -> None:
    config = load_contract(CONFIG)
    exposure, curves, _ = _characteristic_curves(config)
    rng = np.random.default_rng(8122026)
    source = rng.uniform(0.0, 1.0, size=(256, 3))
    truth = np.array([-3.0, 0.2, 0.3, 3.2], dtype=np.float64)
    target = _apply_curve(source, truth, exposure, curves[0])
    actual = _fit_characteristic_layer(source, target, exposure, curves[0], 0)
    prediction = _apply_curve(source, actual, exposure, curves[0])
    assert np.max(np.abs(prediction - target)) < 1e-6
    assert np.all(actual[1:] >= 0.0)


def test_invalid_source_fails_before_curve_application() -> None:
    config = load_contract(CONFIG)
    exposure, curves, _ = _characteristic_curves(config)
    with pytest.raises(FujifilmCharacteristicForwardError, match="application"):
        _apply_curve(
            np.array([[1.1, 0.0, 0.0]]),
            np.array([-3.0, 1.0, 0.0, 0.0]),
            exposure,
            curves[0],
        )


def test_formal_characteristic_forward_proxy_is_repeat_exact_and_passes() -> None:
    contract = load_contract(CONFIG)
    first = evaluate_characteristic_forward_proxy(contract, ROOT)
    second = evaluate_characteristic_forward_proxy(contract, ROOT)
    assert first == second
    assert first["passed"] is True
    assert first["failed_gates"] == []
    assert first["decision"] == "retain_characteristic_constrained_forward_mechanism"
    assert first["comparisons"]["win_fraction_vs_cb5_affine"] > 0.83
    assert first["summaries"]["characteristic"]["median"] < 0.16
