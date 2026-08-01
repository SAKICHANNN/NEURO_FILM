from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_source_adaptive_lut_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq7_fivek_source_adaptive_lut_v1.json"


def test_bq7_contract_is_source_only_and_group_held() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["descriptor"]["source_pixels_only"] is True
    assert payload["model"]["query_target_used_at_inference"] is False
    assert payload["model"]["confirmation_used_for_fit_or_selection"] is False
    assert payload["model"]["lut_basis_rank"] == 8
    assert payload["confirmation_pixels_allowed"] is False


def test_bq7_real_parent_contract_validates_without_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["parent"]["source_only_routing_research_allowed"] is True
    assert validated["nearest"]["selected_family"] == "tone_layout"
    assert validated["mature_config"]["operator"]["grid_size"] == 4
