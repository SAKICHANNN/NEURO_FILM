from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_fresh_boundary_safe_confirmation import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ay3f_fivek_fresh_boundary_safe_confirmation_v1.json"
)


def test_contract_binds_fresh_set_and_unchanged_safe_executor() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["manifest"]["rows"]) == 63
    assert config["safe_executor"]["change_allowed"] is False
    assert config["production_integration_allowed"] is False
