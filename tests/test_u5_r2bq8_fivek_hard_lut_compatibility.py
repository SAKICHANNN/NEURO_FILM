from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_hard_lut_compatibility_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq8_fivek_hard_lut_compatibility_v1.json"


def test_bq8_contract_preserves_hard_source_only_boundary() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["descriptor"]["source_pixels_only"] is True
    assert payload["model"]["query_target_used_at_inference"] is False
    assert payload["model"]["dense_blending_allowed"] is False
    assert payload["confirmation_pixels_allowed"] is False


def test_bq8_real_parent_contract_validates_without_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["parent"]["hard_case_routing_research_allowed"] is True
    assert validated["mature_config"]["operator"]["grid_size"] == 4
