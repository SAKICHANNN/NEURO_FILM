from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.fivek_adaptive_lut_basis_development import (
    FiveKAdaptiveLUTError,
    apply_residual_lut,
    apply_unbounded_residual_safely,
    fit_residual_lut,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2bj0_fivek_adaptive_lut_basis_development_v1.json"
)


def test_contract_is_fail_closed_and_parent_bound() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert [item["rows"] for item in validated["populations"]] == [64, 63, 64]

    drifted = json.loads(CONFIG.read_text(encoding="utf-8"))
    drifted["operator"]["grid_size"] = 5
    with pytest.raises(FiveKAdaptiveLUTError):
        validate_contract(ROOT, drifted)


def test_identity_residual_lut_is_exact() -> None:
    rng = np.random.default_rng(20260731)
    source = rng.random((17, 19, 3), dtype=np.float64)
    residual = np.zeros((4, 4, 4, 3), dtype=np.float64)
    assert np.array_equal(apply_residual_lut(source, residual), source)


def test_fitted_lut_improves_known_smooth_transform() -> None:
    rng = np.random.default_rng(71)
    source = rng.random((96, 80, 3), dtype=np.float64)
    target = source + 0.04 * np.stack(
        (
            source[..., 1] * (1.0 - source[..., 0]),
            source[..., 2] * (1.0 - source[..., 1]),
            source[..., 0] * (1.0 - source[..., 2]),
        ),
        axis=-1,
    )
    fitted = fit_residual_lut(
        source,
        target,
        grid_size=4,
        sample_stride=2,
        identity_shrinkage=0.01,
        smoothness=0.1,
        maximum_absolute_residual=0.25,
    )
    before = float(np.sqrt(np.mean((source - target) ** 2)))
    after = float(
        np.sqrt(np.mean((apply_residual_lut(source, fitted) - target) ** 2))
    )
    assert after < before * 0.35
    assert np.max(np.abs(fitted)) <= 0.25


def test_unbounded_candidate_uses_one_safe_rgb_scale_without_clipping() -> None:
    source = np.asarray([[[0.2, 0.4, 0.8], [0.0, 0.5, 1.0]]])
    candidate = np.asarray([[[1.4, -0.3, 0.9], [-0.5, 1.4, 1.2]]])
    output, scale = apply_unbounded_residual_safely(
        source, candidate, boundary_epsilon=1.0 / 510.0
    )
    assert np.all(output >= 0.0)
    assert np.all(output <= 1.0)
    reconstructed = source + scale[..., None] * (candidate - source)
    assert np.array_equal(output, reconstructed)
    assert np.all(scale < 1.0)

    extreme = np.asarray([[[1e12, -1e12, 1e12]]])
    safe, extreme_scale = apply_unbounded_residual_safely(
        np.asarray([[[0.25, 0.5, 0.75]]]),
        extreme,
        boundary_epsilon=1.0 / 510.0,
    )
    assert np.all(safe > 0.0)
    assert np.all(safe < 1.0)
    assert 0.0 < extreme_scale.item() < 1.0
