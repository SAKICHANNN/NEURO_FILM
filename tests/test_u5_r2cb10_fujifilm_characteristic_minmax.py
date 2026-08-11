from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_minmax import (
    CONTRACT_SHA256,
    FujifilmCharacteristicMinmaxError,
    apply_characteristic_minmax,
    load_contract,
)
from src.eval.fujifilm_characteristic_photographic import _compiled_curve
from src.eval.fujifilm_dye_basis_measured_conformance import hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import _new_boundary_fraction

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb10_fujifilm_characteristic_minmax_v1.json"


def _curve():
    config = load_contract(CONFIG)
    return _compiled_curve(load_cb6(ROOT / config["parents"]["cb6_contract_path"]))


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB10"


def test_contract_drift_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "changed.json"
    path.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmCharacteristicMinmaxError, match="hash drift"):
        load_contract(path)


def test_minmax_coordinate_preserves_boundaries_neutrals_and_channel_order() -> None:
    epsilon = 1.0 / 510.0
    rng = np.random.default_rng(20260811)
    source = rng.random((257, 263, 3), dtype=np.float32)
    source[0, :, :] = np.linspace(0.0, 1.0, 263, dtype=np.float32)[:, None]
    before = source.copy()
    output = apply_characteristic_minmax(
        source, _curve(), strength=0.2, boundary_epsilon=epsilon
    )
    assert np.array_equal(source, before)
    assert _new_boundary_fraction(source, output, epsilon) == 0.0
    assert np.all(np.argsort(source, axis=-1) == np.argsort(output, axis=-1))
    assert np.array_equal(output[0, :, 0], output[0, :, 1])
    assert np.array_equal(output[0, :, 1], output[0, :, 2])


def test_invalid_source_fails_before_execution() -> None:
    with pytest.raises(FujifilmCharacteristicMinmaxError, match="invalid"):
        apply_characteristic_minmax(
            np.array([[[-0.1, 0.0, 0.0]]], dtype=np.float32),
            _curve(),
            strength=0.2,
            boundary_epsilon=1.0 / 510.0,
        )
