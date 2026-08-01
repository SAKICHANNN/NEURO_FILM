from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_source_hard_retrieval_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq2a_fivek_source_hard_retrieval_v1.json"


def test_bq2a_contract_is_hard_source_only_and_confirmation_frozen() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["status"] == "contract_frozen_implementation_ready"
    assert payload["descriptor_families"] == [
        "global_photometric",
        "spatial_photometric",
        "tone_layout",
    ]
    assert payload["descriptor"]["source_pixels_only"] is True
    assert payload["descriptor"]["target_pixels_used"] is False
    assert payload["descriptor"]["operator_parameters_used"] is False
    assert payload["selector"]["dense_blending_allowed"] is False
    assert payload["confirmation_family_selection_allowed"] is False
    assert payload["learned_final_rgb_allowed"] is False
    assert payload["film_or_stock_claim_allowed"] is False


def test_bq2a_real_parent_contract_validates_without_loading_targets() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["oracle"]["automatic_pass"] is True
    assert validated["manifest"]["split_summary"][
        "selection_used_target_or_pixels"
    ] is False
