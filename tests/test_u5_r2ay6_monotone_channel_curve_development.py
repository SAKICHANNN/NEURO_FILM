from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.boundary_safe_neutral_base import (
    apply_boundary_safe_residual,
)
from src.eval.fivek_monotone_channel_curve_development import (
    apply_curve_operator,
    fit_curve_parameters,
    project_curve_parameters,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ay6_monotone_channel_curve_development_v1.json"
)


def test_contract_binds_distinct_curve_family() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["fresh_manifest"]["rows"]) == 63
    assert config["operator"]["parameter_count"] == 15
    assert config["production_integration_allowed"] is False


def test_projection_and_application_are_monotone_endpoint_exact() -> None:
    knots = np.asarray([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0])
    raw = np.asarray(
        [0.9, 0.1, 0.7, 0.3, 0.2] * 3, dtype=np.float64
    )
    projected = project_curve_parameters(
        raw, knots=knots, endpoint_weight=1_000_000.0
    )
    ramp = np.linspace(0.0, 1.0, 257)
    source = np.repeat(ramp[:, None, None], 3, axis=2)
    output = apply_curve_operator(source, projected, knots=knots)
    assert np.array_equal(output[[0, -1]], source[[0, -1]])
    assert np.all(np.diff(output[:, 0, :], axis=0) >= 0.0)


def test_fit_and_shared_safety_preserve_bounds_without_clipping() -> None:
    knots = np.asarray([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0])
    ramp = np.linspace(0.0, 1.0, 1024).reshape(32, 32, 1)
    source = np.repeat(ramp, 3, axis=2)
    target = np.power(source, [0.8, 1.1, 1.3])
    parameters = fit_curve_parameters(
        source,
        target,
        knots=knots,
        sample_stride=1,
        endpoint_weight=1_000_000.0,
    )
    candidate = apply_curve_operator(source, parameters, knots=knots)
    output, scale = apply_boundary_safe_residual(
        source, candidate, boundary_epsilon=1.0 / 510.0
    )
    assert np.all(np.isfinite(output))
    assert np.all((output >= 0.0) & (output <= 1.0))
    assert np.all((scale >= 0.0) & (scale <= 1.0))
