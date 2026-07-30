from src.eval.filmmatch_condition_oracle import (
    evaluate_selective_high_exposure_oracle,
)


def test_selective_oracle_keeps_global_fallback() -> None:
    parent = {
        "folds": [
            {
                "held_group": f"light|ev={ev:+d}",
                "expert_improvement_over_global": 0.2 if ev >= 2 else -0.5,
            }
            for ev in range(-2, 4)
        ]
    }
    config = {
        "experiment_id": "test",
        "selective_policy": {"minimum_exposure_ev": 2},
        "selective_oracle_gate": {
            "minimum_eligible_folds": 2,
            "minimum_eligible_win_fraction": 1.0,
            "minimum_eligible_median_improvement": 0.1,
            "minimum_eligible_worst_improvement": 0.0,
        },
        "disclosure": {},
        "claim_ceiling": "test",
    }
    report = evaluate_selective_high_exposure_oracle(parent, config)
    assert report["selective_oracle_gate_passed"]
    assert report["aggregate"]["eligible_folds"] == 2
    assert report["aggregate"]["fallback_folds"] == 4
