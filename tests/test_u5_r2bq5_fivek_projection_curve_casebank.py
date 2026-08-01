from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_projection_curve_casebank_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq5_fivek_projection_curve_casebank_v1.json"


def test_bq5_contract_reuses_mature_operator_without_selector() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["source_selector_training_allowed"] is False
    assert payload["confirmation_pixels_allowed"] is False
    assert payload["mature_baseline"]["reuse"].startswith(
        "projection directions"
    )
    assert payload["evaluation"][
        "minimum_self_fit_median_style_retention"
    ] == 0.7
    assert payload["production_integration_allowed"] is False


def test_bq5_real_parent_contract_validates_without_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["parent"]["triangular_fit_or_router_rescue_allowed"] is False
    assert validated["mature_decision"]["automatic_pass"] is True
    assert validated["mature_config"]["operator"]["control_point_count"] == 9
