from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_monotone_channel_curve_confirmation import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ay6f_monotone_channel_curve_confirmation_v1.json"
)


def test_contract_binds_untouched_population_and_frozen_alpha() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["manifest"]["rows"]) == 64
    assert config["development"]["expected_full_training_alpha"] == 100.0
    assert config["confirmation_target_use_for_training_allowed"] is False
    assert config["production_integration_allowed"] is False
