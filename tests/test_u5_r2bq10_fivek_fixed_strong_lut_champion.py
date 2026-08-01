from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_fixed_strong_lut_champion_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq10_fivek_fixed_strong_lut_champion_v1.json"


def test_bq10_contract_is_fixed_fit_only_k1() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["router_training_allowed"] is False
    assert payload["selection"]["target_or_validation_used_for_selection"] is False
    assert payload["selection"]["same_camera_group_rows_excluded_from_candidate_score"] is True
    assert payload["confirmation_pixels_allowed"] is False


def test_bq10_real_parent_contract_validates_without_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["mature_config"]["operator"]["grid_size"] == 4
