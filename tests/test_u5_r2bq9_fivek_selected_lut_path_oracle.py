from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_selected_lut_path_oracle_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq9_fivek_selected_lut_path_oracle_v1.json"


def test_bq9_contract_is_evaluator_only_and_fixed_identity() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["router_or_strength_training_allowed"] is False
    assert payload["path"]["selected_case_identity_refit_or_change_allowed"] is False
    assert payload["confirmation_pixels_allowed"] is False
    assert payload["path"]["minimum_target_style_retention"] == 0.7


def test_bq9_real_parent_contract_validates_without_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["report"]["automatic_pass"] is False
    assert validated["parent_config"]["model"]["dense_blending_allowed"] is False
