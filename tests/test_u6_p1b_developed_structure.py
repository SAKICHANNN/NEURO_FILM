from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_developed_structure import (
    evaluate_developed_structure,
    load_contract,
)
from src.film_physics import (
    PhysicalDomain,
    build_bw_silver_context,
    build_colour_dye_cloud_context,
    render_developed_structure,
    render_developed_structure_region,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p1b_developed_structure_reference_v1.json"


def test_colour_and_bw_have_distinct_physical_domains() -> None:
    colour_context = build_colour_dye_cloud_context(
        np.full((3, 4, 3), (0.2, 0.4, 0.8), dtype=np.float64),
        radius_um_cmy=(5.0, 6.0, 7.0),
        mark_optical_density_cmy=(0.3, 0.35, 0.4),
        output_zoom=4,
        output_pixel_pitch_um=8.0,
        monte_carlo_samples=4,
        seed=1,
    )
    bw_context = build_bw_silver_context(
        np.full((3, 4), 0.7, dtype=np.float64),
        radius_um=5.0,
        output_zoom=4,
        output_pixel_pitch_um=8.0,
        monte_carlo_samples=4,
        seed=2,
    )
    colour = render_developed_structure(colour_context)
    bw = render_developed_structure(bw_context)
    assert colour.domain is PhysicalDomain.DEVELOPED_DENSITY
    assert bw.domain is PhysicalDomain.TRANSMITTANCE
    assert np.all(colour.values >= 0.0)
    assert np.all((bw.values > 0.0) & (bw.values <= 1.0))


def test_colour_region_partition_is_exact() -> None:
    context = build_colour_dye_cloud_context(
        np.full((4, 4, 3), (0.2, 0.4, 0.8), dtype=np.float64),
        radius_um_cmy=(5.0, 6.0, 7.0),
        mark_optical_density_cmy=(0.3, 0.35, 0.4),
        output_zoom=4,
        output_pixel_pitch_um=8.0,
        monte_carlo_samples=4,
        seed=3,
    )
    full = render_developed_structure(context).values
    top = render_developed_structure_region(
        context, output_origin_yx=(0, 0), output_shape=(7, 16)
    ).values
    bottom = render_developed_structure_region(
        context, output_origin_yx=(7, 0), output_shape=(9, 16)
    ).values
    assert np.array_equal(np.concatenate([top, bottom]), full)


def test_builders_reject_invalid_sampling_before_random_generation() -> None:
    with pytest.raises(ValueError, match="radii and marks"):
        build_colour_dye_cloud_context(
            np.ones((2, 2, 3)),
            radius_um_cmy=(0.0, 1.0, 1.0),
            mark_optical_density_cmy=(0.3, 0.3, 0.3),
            output_zoom=4,
            output_pixel_pitch_um=8.0,
            monte_carlo_samples=2,
            seed=0,
        )
    with pytest.raises(ValueError, match="samples and seed"):
        build_bw_silver_context(
            np.ones((2, 2)),
            radius_um=5.0,
            output_zoom=4,
            output_pixel_pitch_um=8.0,
            monte_carlo_samples=0,
            seed=0,
        )


def test_frozen_reference_report_repeats_and_passes() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_developed_structure(contract)
    second = evaluate_developed_structure(contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert all(first["decisions"].values())
    assert first["negative_control"]["id"] == "U6.2B-salt-like-bright-speckles"
