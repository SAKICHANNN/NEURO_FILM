from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_target_free_ood_fallback import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ay4_target_free_ood_fallback_diagnostic_v1.json"
)


def test_contract_binds_failed_tail_as_development_only() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["failed_report"]["automatic_pass"] is False
    assert config["routing"]["target_or_oracle_features_allowed"] is False
    assert config["production_integration_allowed"] is False
