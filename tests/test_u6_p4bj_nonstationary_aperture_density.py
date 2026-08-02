from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.nonstationary_aperture_density import (
    NonstationaryApertureDensityError,
    evaluate_nonstationary_aperture_density,
    load_contract,
)
from src.film_physics.aperture_cell_sampler import (
    counter_aperture_scaled_poisson_region,
    sample_nonstationary_aperture_density_region,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bj_nonstationary_aperture_density_v1.json"
DECISION = ROOT / "configs/u6_p4bj_nonstationary_aperture_density_decision_v1.json"


def test_p4bj_contract_is_frozen_before_implementation() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4bj_nonstationary_aperture_density_contract.v1"
    )
    assert payload["candidate"]["patterns"] == [
        "horizontal_ramp",
        "vertical_step",
        "checker_16",
        "highlight_island",
    ]
    assert payload["candidate"]["spatial_topology"] == (
        "independent_measurement_cells_generic_fallback"
    )
    assert payload["candidate"]["realized_field_renormalization_allowed"] is False
    assert payload["candidate"]["correlation_radius_or_nps_claimed"] is False
    assert payload["candidate"]["photographic_render_allowed"] is False
    assert payload["evaluation"]["required_row_count"] == 36
    assert payload["evaluation"]["require_two_byte_identical_reports"] is True


def test_p4bj_decision_binds_formal_result() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    assert payload["decision"] == "retain_nonstationary_aperture_amplitude_reference"
    assert payload["report_sha256"] == (
        "36d5ffe72e43fface0a23b8e1f458c09efec616ae502203fc4999d5d0a300850"
    )
    assert payload["stable_evidence_id"] == (
        "7d4fecdf700f9bb500dd490702c355bec77745cc5882045f39d3a79e621fcf92"
    )
    assert payload["spatial_nps_identified"] is False
    assert payload["photographic_render_allowed"] is False
    assert payload["product_integration_allowed"] is False


def test_nonstationary_sampler_preserves_scalar_counts_and_partitions() -> None:
    shape = (17, 19)
    rate = np.full(shape, 2432.612581852186, dtype=np.float64)
    mark = np.full(shape, 0.00014339502674720291, dtype=np.float64)
    expected = counter_aperture_scaled_poisson_region(
        shape, origin_yx=(0, 0), shape=shape, rate=float(rate[0, 0]), seed=2608021700
    )
    counts, density = sample_nonstationary_aperture_density_region(
        rate, mark, origin_yx=(0, 0), shape=shape, seed=2608021700
    )
    top_counts, top_density = sample_nonstationary_aperture_density_region(
        rate, mark, origin_yx=(0, 0), shape=(7, 19), seed=2608021700
    )
    bottom_counts, bottom_density = sample_nonstationary_aperture_density_region(
        rate, mark, origin_yx=(7, 0), shape=(10, 19), seed=2608021700
    )
    assert np.array_equal(counts, expected)
    assert np.array_equal(np.concatenate([top_counts, bottom_counts]), counts)
    assert np.array_equal(np.concatenate([top_density, bottom_density]), density)
    assert np.all(density > 0.0)


def test_nonstationary_sampler_rejects_invalid_fields_atomically() -> None:
    rate = np.full((3, 4), 10.0, dtype=np.float64)
    mark = np.full((3, 4), 0.1, dtype=np.float64)
    rate[1, 1] = np.nan
    with pytest.raises(ValueError, match="parameters are invalid"):
        sample_nonstationary_aperture_density_region(
            rate, mark, origin_yx=(0, 0), shape=(3, 4), seed=1
        )


def test_contract_rejects_renormalization(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate"]["realized_field_renormalization_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NonstationaryApertureDensityError, match="contract drift"):
        load_contract(path)


def test_nonstationary_aperture_density_report_passes() -> None:
    report = evaluate_nonstationary_aperture_density(load_contract(CONTRACT), ROOT)
    assert report["automatic_pass"] is True
    assert report["stable_evidence_id"] == (
        "7d4fecdf700f9bb500dd490702c355bec77745cc5882045f39d3a79e621fcf92"
    )
    assert report["row_count"] == 36
    assert all(report["gate_results"].values())
