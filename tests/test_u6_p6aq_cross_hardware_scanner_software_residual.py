from pathlib import Path

import numpy as np
import pytest

from src.eval.cross_hardware_scanner_software_residual import (
    _apply,
    _fit,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6aq_cross_hardware_scanner_software_residual_v1.json"


def test_contract_is_frozen_cross_hardware_design() -> None:
    contract = load_contract(CONTRACT)
    assert len(contract["hardware_pairs"]) == 4
    assert contract["gates"]["required_patch_predictions"] == 5280
    assert contract["candidate"]["held_hardware_pixels_allowed_in_fit"] is False


def test_fit_recovers_shared_power_affine_and_safe_execution() -> None:
    rng = np.random.default_rng(7)
    source = rng.uniform(0.1, 0.8, size=(200, 3))
    target = np.power(source, 1.25) @ np.diag([0.8, 0.9, 0.7]) + 0.05
    candidate = load_contract(CONTRACT)["candidate"]
    model = _fit(source, target, candidate)
    output, receipt = _apply(source, model)
    assert model[0] == 1.25
    assert np.max(np.abs(output - target)) < 1e-8
    assert receipt["limited_fraction"] == 0.0


def test_contract_rejects_fit_leakage() -> None:
    import json

    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["candidate"]["held_hardware_pixels_allowed_in_fit"] = True
    scratch = ROOT / "tmp" / "p6aq-invalid.json"
    scratch.parent.mkdir(exist_ok=True)
    scratch.write_text(json.dumps(value), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            load_contract(scratch)
    finally:
        scratch.unlink(missing_ok=True)
