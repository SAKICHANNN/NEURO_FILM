from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.fivek_neutral_base_parameter_pilot import (
    FiveKNeutralBasePilotError,
    _fit_parameters,
    apply_neutral_base,
    source_descriptor,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/u5_r2ay0_fivek_neutral_base_parameter_pilot_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_ay0_contract_binds_source_and_forbids_direct_rgb() -> None:
    config = _config()
    validate_contract(ROOT, config)
    assert config["operator"]["direct_final_rgb_learning_forbidden"] is True
    assert config["evaluation"]["outer_split"].startswith("five-fold")
    assert "expert-to-RAW-visible-shape" in config["decode"]["pair_alignment"]
    assert any("neural" in row for row in config["forbidden_rescues"])


def test_neutral_base_is_endpoint_exact_neutral_and_cube_safe() -> None:
    rng = np.random.default_rng(8)
    source = rng.random((19, 23, 3), dtype=np.float32)
    source[0, 0] = 0.0
    source[0, 1] = 1.0
    identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    assert np.array_equal(apply_neutral_base(source, identity), source)
    neutral = np.repeat(source[..., :1], 3, axis=2)
    output = apply_neutral_base(
        neutral, np.array([1.2, 0.3, 0.0, 0.0, 0.85])
    )
    assert np.max(np.ptp(output, axis=2)) == 0.0
    stressed = apply_neutral_base(
        source, np.array([1.3, 1.0, 0.35, -0.35, 1.0])
    )
    assert float(stressed.min()) >= 0.0
    assert float(stressed.max()) <= 1.0
    assert np.array_equal(stressed[0, 0], np.zeros(3, dtype=np.float32))
    assert np.array_equal(stressed[0, 1], np.ones(3, dtype=np.float32))


def test_synthetic_parameter_fit_recovers_bounded_operator() -> None:
    config = _config()
    rng = np.random.default_rng(19)
    source = rng.uniform(0.02, 0.98, size=(40, 44, 3)).astype(np.float32)
    truth = np.array([1.13, 0.21, 0.08, -0.11, 0.91])
    target = apply_neutral_base(source, truth)
    fitted, success = _fit_parameters(source, target, config)
    assert success
    assert np.max(np.abs(fitted - truth)) < 0.015


def test_descriptor_is_fixed_source_only_and_invalid_input_fails() -> None:
    config = _config()
    source = np.full((7, 9, 3), 0.4, dtype=np.float32)
    descriptor = source_descriptor(source, config)
    assert descriptor.shape == (42,)
    with pytest.raises(FiveKNeutralBasePilotError):
        apply_neutral_base(source, np.ones(4))
