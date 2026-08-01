from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_pairwise_compatibility_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq2b_fivek_pairwise_compatibility_v1.json"


def test_bq2b_contract_is_low_capacity_hard_and_source_only_at_inference() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["model"]["family"] == (
        "closed_form_ridge_ranker_of_within_query_case_error_rank"
    )
    assert payload["model"]["pca_components"] == 32
    assert payload["model"]["query_target_used_at_inference"] is False
    assert payload["model"]["operator_parameters_used_at_inference"] is False
    assert payload["selector"]["dense_blending_allowed"] is False
    assert payload["confirmation_model_selection_allowed"] is False
    assert payload["learned_final_rgb_allowed"] is False
    assert payload["production_integration_allowed"] is False


def test_bq2b_real_parent_contract_validates_without_loading_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["decision"]["reports"]["automatic_pass"] is False
    assert validated["nearest"]["selected_family"] == "tone_layout"
    assert validated["oracle"]["automatic_pass"] is True
