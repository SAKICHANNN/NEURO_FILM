from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_interimage_adjacency import (
    evaluate_interimage_adjacency,
    load_contract,
)
from src.film_physics.interimage_adjacency import (
    InterimageAdjacencyProfile,
    apply_interimage_adjacency,
    apply_interimage_adjacency_row_tiled,
    interimage_adjacency_profile_from_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p5f_interimage_adjacency_v1.json"


def _profile() -> InterimageAdjacencyProfile:
    return interimage_adjacency_profile_from_contract(load_contract(CONTRACT))


def test_profile_rejects_nonzero_row_sum() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="rows must sum to zero"):
        InterimageAdjacencyProfile(
            **{
                **profile.__dict__,
                "coupling_matrix": (
                    (0.12, -0.05, -0.06),
                    (-0.05, 0.12, -0.06),
                    (-0.06, -0.06, 0.12),
                ),
            }
        )


def test_constant_and_neutral_fields_are_exact_identity() -> None:
    profile = _profile()
    constant = np.full((17, 23, 3), 1.25, dtype=np.float64)
    ramp = np.linspace(0.0, 3.0, 23, dtype=np.float64)
    neutral = np.broadcast_to(ramp[None, :, None], (17, 23, 3)).copy()
    assert np.array_equal(apply_interimage_adjacency(constant, profile), constant)
    assert np.array_equal(apply_interimage_adjacency(neutral, profile), neutral)


def test_row_tiling_is_exact() -> None:
    profile = _profile()
    rng = np.random.default_rng(6205)
    density = rng.uniform(0.1, 2.9, size=(79, 61, 3))
    full = apply_interimage_adjacency(density, profile)
    assert np.array_equal(
        full,
        apply_interimage_adjacency_row_tiled(
            density,
            profile,
            tile_rows=13,
        ),
    )


def test_invalid_density_fails_closed() -> None:
    profile = _profile()
    density = np.ones((5, 7, 3), dtype=np.float64)
    density[1, 1, 0] = 3.1
    with pytest.raises(ValueError, match="outside"):
        apply_interimage_adjacency(density, profile)


def test_frozen_synthetic_candidate_passes() -> None:
    report = evaluate_interimage_adjacency(load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert all(report["decisions"].values())
    assert report["metrics"]["remote_impulse_maximum_absolute_correction"] == 0.0
    assert report["metrics"]["hard_clip_fraction"] == 0.0
