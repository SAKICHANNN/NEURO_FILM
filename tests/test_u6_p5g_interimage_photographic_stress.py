from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_interimage_photographic_stress import (
    _combined_pipeline,
    _combined_pipeline_row_tiled,
    _crop_bounds,
    load_contract,
)
from src.film_physics import (
    SpatialResponseProfile,
    apply_bounded_development_adjacency,
)
from src.film_physics.interimage_adjacency import (
    interimage_adjacency_profile_from_contract,
)
from src.eval.physical_interimage_adjacency import (
    load_contract as load_p5f_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p5g_interimage_photographic_stress_v1.json"
P5F = ROOT / "configs" / "u6_p5f_interimage_adjacency_v1.json"


class _IdentityOperator:
    @staticmethod
    def apply(values: np.ndarray) -> np.ndarray:
        return values + 0.5


def _profile() -> SpatialResponseProfile:
    return SpatialResponseProfile(
        pixel_pitch_um=1.0,
        forward_scatter_sigma_um_rgb=(1.0, 1.0, 1.0),
        development_adjacency_sigma_um_rgb=(1.0, 1.0, 1.0),
        development_adjacency_gain_rgb=(0.18, 0.22, 0.26),
        dye_diffusion_sigma_um_rgb=(1.0, 1.2, 1.4),
        scanner_mtf_sigma_um_rgb=(0.8, 0.9, 1.0),
    )


def _p5c_apply():
    return partial(
        apply_bounded_development_adjacency,
        maximum_absolute_transmittance_delta=0.008,
        maximum_absolute_density_delta=0.08,
    )


def test_p5g_contract_reuses_frozen_cohort_and_partitions() -> None:
    contract = load_contract(CONTRACT)
    assert contract["input"]["expected_rows"] == 18
    assert contract["input"]["expected_camera_makes"] == 9
    assert contract["pipeline"]["row_partitions"] == [257, 509]


def test_combined_pipeline_row_tiling_is_exact() -> None:
    rng = np.random.default_rng(6207)
    exposure = rng.uniform(0.0, 1.0, size=(713, 83, 3))
    profile = _profile()
    p5f_profile = interimage_adjacency_profile_from_contract(
        load_p5f_contract(P5F)
    )
    full = _combined_pipeline(
        exposure,
        profile,
        _IdentityOperator(),
        _p5c_apply(),
        p5f_profile,
    )
    tiled = _combined_pipeline_row_tiled(
        exposure,
        profile,
        _IdentityOperator(),
        _p5c_apply(),
        p5f_profile,
        tile_rows=257,
    )
    assert np.array_equal(full, tiled)


def test_combined_pipeline_rejects_invalid_partition() -> None:
    density = np.ones((5, 7, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="positive integer"):
        _combined_pipeline_row_tiled(
            density,
            _profile(),
            _IdentityOperator(),
            _p5c_apply(),
            interimage_adjacency_profile_from_contract(
                load_p5f_contract(P5F)
            ),
            tile_rows=0,
        )


def test_crop_bounds_stay_inside_image() -> None:
    ys, xs = _crop_bounds((300, 400), (299, 399), 256)
    assert (ys.start, ys.stop) == (44, 300)
    assert (xs.start, xs.stop) == (144, 400)
