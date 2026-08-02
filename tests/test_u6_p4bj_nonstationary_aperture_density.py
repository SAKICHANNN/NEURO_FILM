from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bj_nonstationary_aperture_density_v1.json"


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
