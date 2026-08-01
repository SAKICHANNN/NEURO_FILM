from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_operator_style_ceiling_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq4d_fivek_operator_style_ceiling_v1.json"


def test_bq4d_contract_is_diagnostic_only() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["training_or_fitting_allowed"] is False
    assert payload["confirmation_pixels_allowed"] is False
    assert payload["safe_execution"]["hard_clipping_allowed"] is False
    assert payload["diagnostic_gates"][
        "minimum_self_fit_median_style_retention"
    ] == 0.7
    assert payload["product_integration_allowed"] is False


def test_bq4d_real_parent_contract_validates_without_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["bq4"]["confirmation_executed"] is False
    assert validated["oracle"]["automatic_pass"] is True
    assert validated["decision"]["threshold_or_capacity_rescue_allowed"] is False
