from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.response_bounded_fresh_confirmation import load_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3o_response_bounded_fresh_confirmation_v1.json"


def test_p3o_contract_is_fresh_fixed_and_nonfitting() -> None:
    contract = load_contract(CONTRACT)
    assert contract["node"] == "U6.P3O"
    assert contract["population"]["eligible_source_count"] == 17
    assert contract["population"]["camera_make_count"] == 9
    assert contract["population"]["excluded_ids"] == ["fujifilm_s2pro"]
    assert contract["population"][
        "require_zero_exact_raw_hash_overlap_with_development"
    ]
    assert not contract["population"]["selection_used_image_appearance"]
    assert contract["candidate"]["maximum_transmittance_delta"] == 0.009
    assert contract["candidate"]["shared_rgb_residual_scale"]
    assert not contract["candidate"]["profile_fitting_allowed"]
    assert not contract["candidate"]["per_image_parameter_selection_allowed"]
    assert not contract["training_allowed"]
    assert not contract["production_integration_allowed"]


def test_p3o_contract_preserves_development_gates() -> None:
    contract = load_contract(CONTRACT)
    gates = contract["automatic_gates"]
    assert gates["bounded_maximum_abs_maximum"] == 0.009000000001
    assert gates["bounded_population_p95_abs_minimum"] == 0.0075
    assert (
        gates["bounded_median_fraction_pixels_over_0p005_minimum"] == 0.3
    )
    assert gates["bounded_median_shared_scale_minimum"] == 0.8
    assert gates["maximum_isolated_candidate_excursion_count"] == 0
    visual = contract["visual_protocol"]
    assert len(visual["fixed_ids"]) == 9
    assert len(visual["blind_value_round_seeds"]) == 3
    assert visual["minimum_candidate_votes_per_round"] == 5
    assert visual["minimum_passing_rounds"] == 2


def test_p3o_contract_rejects_schema_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["schema"] = "wrong"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_contract(path)
