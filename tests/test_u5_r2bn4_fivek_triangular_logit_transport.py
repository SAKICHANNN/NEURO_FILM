from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fivek_triangular_logit_transport_development import (
    FiveKTriangularTransportError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2bn4_fivek_triangular_logit_transport_development_v1.json"
)


def test_contract_binds_parent_data_and_populations() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["lower_bounds"].shape == (14,)
    assert validated["upper_bounds"].shape == (14,)
    assert [row["rows"] for row in validated["populations"]] == [64, 63, 64]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("operator", "hard_output_clipping_allowed", True),
        ("operator", "spatial_or_semantic_features_allowed", True),
        ("operator", "learned_final_rgb_allowed", True),
        ("basis_prediction", "basis_rank", 9),
    ],
)
def test_contract_rejects_frozen_boundary_drift(
    section: str, key: str, value: object
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config[section][key] = value
    with pytest.raises(FiveKTriangularTransportError):
        validate_contract(ROOT, config)


def test_contract_rejects_parameter_bound_drift() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["operator"]["lower_bounds"][0] = 0.0
    with pytest.raises(FiveKTriangularTransportError):
        validate_contract(ROOT, config)
