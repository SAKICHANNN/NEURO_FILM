from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fivek_triangular_logit_transport_visual_product_value import (
    ARMS,
    FiveKTriangularVisualError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2bn6_triangular_logit_transport_visual_product_value_v1.json"
)


def test_visual_contract_binds_confirmed_model_and_independent_rows() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert tuple(config["render_arms"]) == ARMS
    assert len(validated["eligible_ids"]) == 12
    assert config["rendering"]["operator_and_ao6_resolution"] == (
        "exact decoded source pixels"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("frozen_model", "target_pixels_available"), True),
        (("rendering", "operator_or_ao6_resampling_allowed"), True),
        (("automatic_gate", "boundary_epsilon"), 0.0),
        (("blind_protocol", "rounds"), 4),
    ],
)
def test_visual_contract_rejects_policy_drift(
    path: tuple[str, str], value: object
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config[path[0]][path[1]] = value
    with pytest.raises(FiveKTriangularVisualError):
        validate_contract(ROOT, config)
