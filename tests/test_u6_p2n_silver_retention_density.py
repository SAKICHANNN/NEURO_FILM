from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_silver_retention import (
    evaluate_silver_retention,
    load_contract,
)
from src.film_physics.silver_retention import (
    SilverRetentionProfile,
    apply_silver_retention,
    apply_silver_retention_row_tiled,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p2n_silver_retention_density_v1.json"


def _profile(fraction: float = 0.35) -> SilverRetentionProfile:
    return SilverRetentionProfile(
        fraction, (0.2126, 0.7152, 0.0722), 3.0, 4.05
    )


def test_zero_retention_is_exact_identity() -> None:
    rng = np.random.default_rng(8)
    density = rng.uniform(0.0, 3.0, size=(17, 23, 3)).astype(np.float32)
    result = apply_silver_retention(density, _profile(0.0))
    assert np.array_equal(result.total_density, density)
    assert np.array_equal(
        result.silver_density, np.zeros(density.shape[:2], dtype=np.float32)
    )


def test_partition_and_input_are_exact() -> None:
    rng = np.random.default_rng(9)
    density = rng.uniform(0.0, 3.0, size=(37, 41, 3)).astype(np.float64)
    before = density.tobytes()
    full = apply_silver_retention(density, _profile())
    tiled = apply_silver_retention_row_tiled(
        density, _profile(), tile_rows=11
    )
    assert np.array_equal(full.total_density, tiled.total_density)
    assert np.array_equal(full.silver_density, tiled.silver_density)
    assert density.tobytes() == before


def test_invalid_density_fails_without_clipping() -> None:
    density = np.zeros((2, 2, 3), dtype=np.float64)
    density[0, 0, 1] = 3.1
    with pytest.raises(ValueError, match="bounded"):
        apply_silver_retention(density, _profile())


def test_formal_evaluator_is_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_silver_retention(contract)
    second = evaluate_silver_retention(contract)
    assert first == second
    assert first["stable_evidence_id"] == second["stable_evidence_id"]
