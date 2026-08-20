from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2spcp2_global_logit_affine_preference_d0_v1.json"


def test_spcp2_contract_freezes_scene_disjoint_roles_before_pixels() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "U5.R2SPCP2"
    assert payload["selection"]["fit_scenes"] == 96
    assert payload["selection"]["calibration_scenes"] == 32
    assert payload["selection"]["sealed_scenes"] == 32
    assert payload["selection"]["replacement_forbidden"] is True
    assert payload["acquisition"]["initial_member_count_exact"] == 256
    assert payload["acquisition"][
        "sealed_image_payload_read_before_development_pass_forbidden"
    ] is True


def test_spcp2_contract_is_one_source_free_explicit_operator() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    fit = payload["fit"]
    assert "one shared 3x3 M" in fit["operator"]
    assert fit["application_source_used_for_fit_or_selection"] is False
    assert fit["output_is_intrinsically_bounded"] is True
    assert "per-scene routing" in payload["stop_rules"][3]


def test_spcp2_contract_freezes_discriminating_controls_and_tails() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert len(payload["controls"]) == 4
    metrics = payload["calibration_metrics"]
    assert metrics["candidate_improvement_rate_min"] == 0.75
    assert metrics["candidate_improvement_worst_min"] == -0.10
    assert metrics["reverse_direction_improvement_rate_max"] == 0.35
    assert metrics["new_exact_boundary_fraction_max"] == 0.0
