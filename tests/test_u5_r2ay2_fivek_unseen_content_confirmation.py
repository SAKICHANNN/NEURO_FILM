from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.fivek_neutral_base_parameter_pilot import (
    fit_neutral_base_parameters,
)
from src.eval.fivek_unseen_content_confirmation import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ay2_fivek_unseen_content_confirmation_v1.json"
)


def test_contract_binds_passed_source_and_frozen_alpha() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["confirmation_manifest"]["rows"]) == 64
    assert config["development"]["ridge_alpha"] == 100.0
    assert config["confirmation"]["group_claim"].startswith(
        "unseen content only"
    )


def test_public_fit_function_preserves_bounded_identity_fit() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    development = json.loads(
        (ROOT / config["development"]["config"]).read_text(encoding="utf-8")
    )
    source = np.full((8, 9, 3), 0.4, dtype=np.float64)
    parameters, success = fit_neutral_base_parameters(
        source, source.copy(), development
    )
    assert success
    assert np.all(parameters >= development["operator"]["lower_bounds"])
    assert np.all(parameters <= development["operator"]["upper_bounds"])
