from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.ao6_density_grain_value import (
    AO6DensityGrainError,
    apply_linear_density_grain,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "u5_r2bc1_ao6_density_grain_value_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_density_grain_preserves_linear_chromaticity_and_cube() -> None:
    y, x = np.mgrid[0:64, 0:96]
    base = np.stack(
        [
            0.05 + 0.9 * x / 95,
            0.08 + 0.7 * y / 63,
            np.full_like(x, 0.35, dtype=np.float64),
        ],
        axis=2,
    ).astype(np.float32)
    output, metrics = apply_linear_density_grain(
        base, density_sigma=0.0225, seed=7
    )
    assert output.shape == base.shape
    assert output.min() >= 0.0
    assert output.max() <= 1.0
    assert metrics["new_raw_clipping_fraction"] == 0.0
    assert metrics["linear_chromaticity_drift_p999"] < 2e-6
    assert metrics["changed_pixel_fraction"] > 0.5


def test_density_grain_is_exact_for_same_seed_and_changes_for_new_seed() -> None:
    base = np.full((31, 47, 3), [0.2, 0.4, 0.7], dtype=np.float32)
    first, first_metrics = apply_linear_density_grain(
        base, density_sigma=0.0225, seed=11
    )
    second, second_metrics = apply_linear_density_grain(
        base, density_sigma=0.0225, seed=11
    )
    third, _ = apply_linear_density_grain(
        base, density_sigma=0.0225, seed=12
    )
    assert np.array_equal(first, second)
    assert first_metrics == second_metrics
    assert not np.array_equal(first, third)


def test_density_grain_rejects_invalid_input() -> None:
    with pytest.raises(AO6DensityGrainError, match="invalid"):
        apply_linear_density_grain(
            np.full((4, 4, 3), np.nan, dtype=np.float32),
            density_sigma=0.0225,
            seed=1,
        )


def test_frozen_contract_binds_parent_and_population() -> None:
    rows = validate_contract(ROOT, _config())
    assert len(rows) == 16
