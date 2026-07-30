from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.boundary_safe_neutral_base import (
    apply_boundary_safe_neutral_base,
    validate_contract,
)
from src.eval.fivek_neutral_base_parameter_pilot import apply_neutral_base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ay3_boundary_safe_neutral_base_development_v1.json"
)


def test_contract_binds_closed_ay2_as_seen_development() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["ay2_report"]["automatic_pass"] is False
    assert config["development_evidence"]["status"].startswith(
        "seen development"
    )


def test_safe_executor_uses_one_shared_scale_and_blocks_new_boundary() -> None:
    epsilon = 1.0 / 510.0
    source = np.array(
        [[[0.5, 0.6, 0.997], [0.2, 0.3, 0.4]]], dtype=np.float64
    )
    parameters = np.array([1.8, 1.5, 0.4, -0.2, 1.0])
    output, scale = apply_boundary_safe_neutral_base(
        source, parameters, boundary_epsilon=epsilon
    )
    assert scale.shape == source.shape[:2]
    residual = output - source
    raw_residual = apply_neutral_base(source, parameters) - source
    nonzero = np.abs(raw_residual[0, 0]) > 1e-12
    ratios = residual[0, 0, nonzero] / raw_residual[0, 0, nonzero]
    assert np.allclose(ratios, scale[0, 0])
    source_boundary = (source <= epsilon) | (source >= 1.0 - epsilon)
    output_boundary = (output <= epsilon) | (output >= 1.0 - epsilon)
    assert not np.any(output_boundary & ~source_boundary)


def test_safe_executor_is_identity_exact() -> None:
    source = np.linspace(0.0, 1.0, 30).reshape(2, 5, 3)
    identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    output, scale = apply_boundary_safe_neutral_base(
        source, identity, boundary_epsilon=1.0 / 510.0
    )
    assert np.array_equal(output, source)
    assert np.array_equal(scale, np.ones(source.shape[:2]))
