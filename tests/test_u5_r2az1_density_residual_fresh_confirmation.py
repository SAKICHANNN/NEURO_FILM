from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_az1_contract_freezes_exact_candidate_and_fresh_population() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2az1_density_residual_fresh_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["status"] == "contract_frozen_before_candidate_render"
    assert config["candidate"]["neutral_strength"] == 0.15
    assert config["candidate"]["opponent_strength"] == 0.35
    assert config["fresh_population"]["required_sample_count"] == 16
    assert config["fresh_population"]["required_camera_make_count"] == 9
    assert (
        config["fresh_population"][
            "required_zero_decoded_sha256_overlap_with_az0_development"
        ]
        is True
    )
    assert config["visual_protocol"][
        "minimum_candidate_preferences_per_passing_round"
    ] == 6
    assert config["visual_protocol"]["minimum_passing_rounds"] == 2
    assert config["operator_refit_allowed"] is False
    assert config["strength_retuning_allowed"] is False
    assert config["production_integration_allowed"] is False
