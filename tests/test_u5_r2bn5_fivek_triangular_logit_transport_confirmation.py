from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fivek_hard_case_medoid_development import _load_fresh_population
from src.eval.fivek_triangular_logit_transport_confirmation import (
    FiveKTriangularConfirmationError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2bn5_fivek_triangular_logit_transport_confirmation_v1.json"
)


def test_confirmation_is_disjoint_and_threshold_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["pair_overlap"] == []
    assert validated["hash_overlap"] == []
    assert len(validated["confirmation_manifest"]["rows"]) == 64


def test_confirmation_rejects_target_access_and_threshold_relaxation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["leakage_controls"][
        "target_pixels_allowed_for_adaptive_prediction"
    ] = True
    with pytest.raises(FiveKTriangularConfirmationError):
        validate_contract(ROOT, config)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["evaluation"]["maximum_inverse_roundtrip_error"] = 1.0e-6
    with pytest.raises(FiveKTriangularConfirmationError):
        validate_contract(ROOT, config)


def test_confirmation_loader_does_not_invent_camera_metadata() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    ay0_config = validated["parent_validated"]["bj0"]["curve_validated"][
        "ay0_config"
    ]
    population = _load_fresh_population(
        ROOT,
        ay0_config,
        {"rows": validated["confirmation_manifest"]["rows"][:1]},
        group_field=None,
    )
    assert population["rows"][0]["group"] == "unknown"
