from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_characteristic_forward_proxy import (
    load_contract as load_cb6_contract,
)
from src.eval.fujifilm_characteristic_photographic import _compiled_curve
from src.eval.fujifilm_characteristic_rgb import (
    CONTRACT_SHA256,
    FujifilmCharacteristicRgbError,
    _minimum_derivative,
    apply_characteristic_rgb,
    load_contract,
)
from src.eval.fujifilm_dye_basis_measured_conformance import hash_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb8_fujifilm_characteristic_rgb_v1.json"
DECISION = ROOT / "configs/u5_r2cb8_fujifilm_characteristic_rgb_decision_v1.json"


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB8"


def test_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmCharacteristicRgbError, match="hash drift"):
        load_contract(changed)


def test_intrinsic_rgb_curve_preserves_cube_neutral_and_order() -> None:
    contract = load_contract(CONFIG)
    parent = load_cb6_contract(ROOT / contract["parents"]["cb6_contract_path"])
    curve = _compiled_curve(parent)
    ramp = np.linspace(0.0, 1.0, 65537, dtype=np.float32)
    source = np.repeat(ramp[:, None], 3, axis=1)
    output = apply_characteristic_rgb(
        source, curve, strength=float(contract["operator"]["strength"])
    )
    assert output[0, 0] == 0.0
    assert output[-1, 0] == 1.0
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.array_equal(output[:, 0], output[:, 1])
    assert np.array_equal(output[:, 1], output[:, 2])
    assert np.all(np.diff(output[:, 0]) > 0.0)


def test_analytic_derivative_meets_frozen_lower_bound() -> None:
    contract = load_contract(CONFIG)
    parent = load_cb6_contract(ROOT / contract["parents"]["cb6_contract_path"])
    curve = _compiled_curve(parent)
    minimum = _minimum_derivative(
        curve,
        strength=float(contract["operator"]["strength"]),
        count=int(contract["evaluation"]["derivative_probe_count"]),
    )
    assert minimum >= float(contract["automatic_gates"]["minimum_curve_derivative"])


def test_invalid_source_fails_before_rgb_execution() -> None:
    contract = load_contract(CONFIG)
    parent = load_cb6_contract(ROOT / contract["parents"]["cb6_contract_path"])
    curve = _compiled_curve(parent)
    with pytest.raises(FujifilmCharacteristicRgbError, match="invalid"):
        apply_characteristic_rgb(
            np.array([[[-0.1, 0.0, 0.0]]], dtype=np.float32),
            curve,
            strength=0.2,
        )


def test_formal_decision_binds_exact_failed_replay() -> None:
    import json

    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["decision"] == "close_intrinsic_rgb_characteristic_without_rescue"
    assert decision["repeat_report_byte_exact"] is True
    assert decision["formal_report_sha256"] == (
        "2510e9a9afb03fdb34f954cc5d59984f34cbd61899e158b8e7e6715ea15c4f20"
    )
    assert decision["failed_gates"] == ["gradient_order", "new_boundaries"]
