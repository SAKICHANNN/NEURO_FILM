from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.fivek_hard_case_medoid_development import (
    _parameter_choices,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ay5_hard_case_medoid_development_v1.json"
)


def test_contract_binds_closed_uncertainty_and_discrete_cases() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["fresh_manifest"]["rows"]) == 63
    assert config["fixed_model"]["case_blending_allowed"] is False
    assert config["production_integration_allowed"] is False


def test_parameter_choices_return_actual_training_cases() -> None:
    train_x = np.arange(30, dtype=np.float64).reshape(6, 5)
    train_y = np.arange(30, dtype=np.float64).reshape(6, 5) / 100
    train_ids = np.asarray([f"case-{index}" for index in range(6)])
    choices, identities = _parameter_choices(
        train_x,
        train_y,
        train_ids,
        train_x[[1, 4]] + 0.01,
        alpha=100.0,
        lower=np.full(5, -10.0),
        upper=np.full(5, 10.0),
    )
    for method in (
        "content_nearest",
        "ridge_projected_medoid",
        "content_top3_ridge_medoid",
        "content_top5_ridge_medoid",
    ):
        assert all(value in train_ids for value in identities[method])
        for row in choices[method]:
            assert any(np.array_equal(row, candidate) for candidate in train_y)
