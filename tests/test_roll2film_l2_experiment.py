from __future__ import annotations

from scripts.run_roll2film_e0_l2_fixed_budget import run


def _config() -> dict[str, object]:
    return {
        "seed": 77,
        "group_sizes": [1, 4],
        "total_target_pixels": 256,
        "neutral_prior_pixels": 512,
        "holdout_pixels": 256,
        "operator_grid_size": 5,
        "estimator_iterations": 2,
        "replicates": 3,
        "bootstrap_resamples": 100,
        "nuisance": {
            "exposure_sigma": 0.12,
            "white_balance_sigma": 0.04,
            "scene_mean_sigma": 0.01,
            "sensor_noise_sigma": 0.001,
        },
        "prior_swap_offset": [0.03, -0.02, 0.01],
        "gate": {
            "gate_group_size": 4,
            "partition_operator_grid_max_abs": 1e-12,
            "paired_relative_improvement_min": 0.1,
        },
    }


def test_l2_fixed_budget_experiment_is_deterministic_and_keeps_claim_boundary() -> None:
    first = run(_config())
    second = run(_config())

    assert first == second
    assert first["partition_equivalence"]["operator_grid_max_abs"] == 0.0
    assert first["operator_contract"]["jacobian_min_on_grid"] > 0.0
    assert first["operator_contract"]["lut65_max_abs"] < first["operator_contract"]["lut33_max_abs"]
    assert first["decision"]["roll_information_decision"] == "not_established"
