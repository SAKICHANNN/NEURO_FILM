from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_characteristic_forward_proxy import (
    load_contract as load_cb6_contract,
)
from src.eval.fujifilm_characteristic_photographic import (
    CONTRACT_SHA256,
    FujifilmCharacteristicPhotographicError,
    _compiled_curve,
    _gamma_residual,
    apply_characteristic_tone,
    load_contract,
)
from src.eval.fujifilm_dye_basis_measured_conformance import hash_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb7_fujifilm_characteristic_photographic_v1.json"


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB7"


def test_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmCharacteristicPhotographicError, match="hash drift"):
        load_contract(changed)


def test_compiled_curve_is_endpoint_exact_monotone_and_nontrivial() -> None:
    contract = load_contract(CONFIG)
    parent = load_cb6_contract(ROOT / contract["parent"]["contract_path"])
    curve = _compiled_curve(parent)
    coordinate = np.linspace(0.0, 1.0, 1001)
    values = curve(coordinate)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert np.all(np.diff(values) >= 0.0)
    assert abs(float(values[500]) - 0.5) > 0.4


def test_characteristic_tone_preserves_neutral_axis_and_bounds() -> None:
    contract = load_contract(CONFIG)
    parent = load_cb6_contract(ROOT / contract["parent"]["contract_path"])
    curve = _compiled_curve(parent)
    ramp = np.linspace(0.0, 1.0, 513, dtype=np.float32)
    source = np.repeat(ramp[:, None, None], 3, axis=2)
    output, source_lab, output_lab, scale = apply_characteristic_tone(
        source, curve, strength=float(contract["operator"]["strength"])
    )
    assert np.all(np.isfinite(output))
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.max(np.abs(output[..., 0] - output[..., 1])) < 2e-6
    assert np.max(np.abs(output[..., 1] - output[..., 2])) < 2e-6
    assert np.all(np.diff(output_lab[..., 0].reshape(-1)) >= -1e-5)
    assert np.all(scale >= 0.0)
    assert np.all(source_lab[..., 0] >= 0.0)


def test_gamma_control_recovers_exact_gamma() -> None:
    source = np.linspace(0.0, 100.0, 4097, dtype=np.float64)
    target = np.power(source / 100.0, 1.3) * 100.0
    indices = np.arange(source.size, dtype=np.int64)
    gamma, residual = _gamma_residual(source, target, indices, (0.25, 4.0))
    assert abs(gamma - 1.3) < 1e-6
    assert residual < 1e-5


def test_invalid_source_fails_before_tone_execution() -> None:
    contract = load_contract(CONFIG)
    parent = load_cb6_contract(ROOT / contract["parent"]["contract_path"])
    curve = _compiled_curve(parent)
    with pytest.raises(FujifilmCharacteristicPhotographicError, match="invalid"):
        apply_characteristic_tone(
            np.array([[[1.1, 0.0, 0.0]]], dtype=np.float32),
            curve,
            strength=0.2,
        )
