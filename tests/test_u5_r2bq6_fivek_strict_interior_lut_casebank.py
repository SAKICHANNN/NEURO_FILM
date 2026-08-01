from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_strict_interior_lut_casebank_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq6_fivek_strict_interior_lut_casebank_v1.json"


def test_bq6_contract_reuses_mature_operator_without_selector() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["source_selector_training_allowed"] is False
    assert payload["confirmation_pixels_allowed"] is False
    assert payload["evaluation"]["minimum_self_fit_median_style_retention"] == 0.7
    assert payload["production_integration_allowed"] is False


def test_bq6_real_parent_contract_validates_without_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["parent"]["projection_curve_fit_or_selector_rescue_allowed"] is False
    assert validated["mature_decision"]["status"] == "closed_fresh_p95_tail_failed"
    assert validated["mature_config"]["operator"]["grid_size"] == 4
