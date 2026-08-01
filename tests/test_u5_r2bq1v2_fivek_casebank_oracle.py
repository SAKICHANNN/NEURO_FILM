from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq1v2_fivek_casebank_oracle_v1.json"


def test_bq1v2_contract_freezes_oracle_and_product_boundaries() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["status"] == "contract_frozen_implementation_ready"
    assert payload["parent_population"]["development_rows"] == 381
    assert payload["parent_population"]["confirmation_rows"] == 128
    assert payload["parent_source_adjudication"]["required_automatic_pass"] is True
    assert payload["required_pass_variants"] == ["aligned_expert", "filtered"]
    assert payload["router_training_allowed"] is False
    assert payload["confirmation_target_use_for_selection_allowed"] is False
    assert payload["final_rgb_learning_allowed"] is False
    assert payload["film_or_stock_claim_allowed"] is False


def test_bq1v2_contract_freezes_operator_controls_and_gates() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    operator = payload["operator"]
    assert operator["family"] == "monotone_triangular_logit_transport"
    assert operator["parameter_count"] == 14
    assert operator["hard_output_clipping_allowed"] is False
    assert operator["learned_final_rgb_allowed"] is False
    expected_doses = [index / 16 for index in range(17)]
    for variant in payload["target_variants"].values():
        evaluation = variant["evaluation"]
        assert evaluation["strength_doses"] == expected_doses
        assert evaluation["bootstrap_repetitions"] == 2000
        assert evaluation["gates"] == {
            "minimum_mean_improvement_over_identity": 0.05,
            "minimum_win_fraction_over_identity": 0.6,
            "maximum_p95_ratio_to_identity": 1.0,
            "maximum_worst_ratio_to_identity": 1.1,
            "minimum_mean_improvement_over_global": 0.05,
            "minimum_win_fraction_over_global": 0.6,
            "maximum_p95_ratio_to_global": 1.0,
            "maximum_worst_ratio_to_global": 1.1,
            "minimum_mean_improvement_over_strength_oracle": 0.03,
            "minimum_win_fraction_over_strength_oracle": 0.55,
            "minimum_mean_improvement_over_random_case": 0.05,
            "minimum_win_fraction_over_random_case": 0.6,
            "minimum_bootstrap_lower_improvement": 0.02,
            "minimum_distinct_selected_cases": 8,
            "maximum_selected_case_share": 0.2,
        }
