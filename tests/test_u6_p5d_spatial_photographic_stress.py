from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_spatial_photographic_stress import (
    _isolated_excursions,
    load_contract,
)
from src.film_physics import (
    SpatialResponseProfile,
    apply_spatial_response_pipeline,
    apply_spatial_response_pipeline_row_tiled,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p5d_spatial_photographic_stress_v1.json"


def _profile() -> SpatialResponseProfile:
    return SpatialResponseProfile(
        pixel_pitch_um=1.0,
        forward_scatter_sigma_um_rgb=(1.0, 1.0, 1.0),
        development_adjacency_sigma_um_rgb=(1.0, 1.0, 1.0),
        development_adjacency_gain_rgb=(0.0, 0.0, 0.0),
        dye_diffusion_sigma_um_rgb=(1.0, 1.0, 1.0),
        scanner_mtf_sigma_um_rgb=(1.0, 1.0, 1.0),
    )


def test_p5d_contract_is_frozen() -> None:
    contract = load_contract(CONTRACT)
    assert contract["input"]["expected_rows"] == 18
    assert contract["pipeline"]["row_partitions"] == [257, 509]


def test_row_partition_pipeline_is_exact() -> None:
    rng = np.random.default_rng(20260728)
    exposure = rng.random((700, 96, 3), dtype=np.float64)
    profile = _profile()
    full = apply_spatial_response_pipeline(
        exposure,
        profile,
        sensitometry_apply=lambda values: values + 0.2,
    )
    tiled = apply_spatial_response_pipeline_row_tiled(
        exposure,
        profile,
        sensitometry_apply=lambda values: values + 0.2,
        tile_rows=257,
    )
    assert np.array_equal(full, tiled)


def test_isolated_excursion_detector_separates_points_from_edges() -> None:
    isolated = np.zeros((16, 16, 3), dtype=np.float64)
    isolated[8, 8] = 0.02
    assert (
        _isolated_excursions(
            isolated, threshold=0.008, radius=2, minimum_support=3
        )
        == 1
    )
    edge = np.zeros((16, 16, 3), dtype=np.float64)
    edge[8, 5:11] = 0.02
    assert (
        _isolated_excursions(
            edge, threshold=0.008, radius=2, minimum_support=3
        )
        == 0
    )
