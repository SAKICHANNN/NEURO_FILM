from __future__ import annotations

from copy import deepcopy

from scripts.run_u5_r2s1_histogram_score_confirmation import (
    evaluate_confirmation_gates,
)


def _method_metrics(error: float) -> dict:
    return {
        "operator_output_rmse": {"median": error, "p90": 0.05},
        "velocity_direction_cosine": {"median": 0.95},
        "identity_rmse_retention_ratio": {"median": 0.98},
        "best_affine_residual_retention_ratio": {"median": 0.9},
        "query_palette_log_density_gain": {"median": 2.0},
        "reference_separation_ratio": {"median": 0.9},
        "structure": {
            "minimum_output": 0.1,
            "maximum_output": 0.9,
            "minimum_jacobian_determinant": 0.01,
            "maximum_jacobian_spectral_norm": 4.0,
            "maximum_inverse_error": 1e-6,
            "maximum_replay_error": 0.0,
            "maximum_coefficient_vector_norm": 1.9,
            "permutation_operator_error": 0.0,
        },
    }


def _gates() -> dict:
    return {
        "maximum_operator_output_rmse_median": 0.04,
        "maximum_operator_output_rmse_p90": 0.065,
        "minimum_velocity_direction_cosine_median": 0.9,
        "minimum_identity_rmse_retention_ratio_median": 0.8,
        "maximum_identity_rmse_retention_ratio_median": 1.2,
        "minimum_best_affine_residual_retention_ratio_median": 0.75,
        "minimum_query_palette_log_density_gain_median": 1.5,
        "minimum_reference_separation_ratio_median": 0.8,
        "minimum_median_error_improvement_over_hard_1nn_fraction": 0.45,
        "minimum_median_error_improvement_over_global_fraction": 0.5,
        "maximum_histogram_permutation_error": 0.0,
        "maximum_operator_permutation_error": 0.0,
        "minimum_output": 0.0,
        "maximum_output": 1.0,
        "minimum_jacobian_determinant_exclusive": 0.005,
        "maximum_jacobian_spectral_norm": 8.0,
        "maximum_inverse_error": 1e-5,
        "maximum_replay_error": 0.0,
        "maximum_coefficient_vector_norm": 2.0,
    }


def _methods() -> dict:
    return {
        "query_kde_0p12": {"metrics": _method_metrics(0.03)},
        "hard_1nn_hellinger": {"metrics": _method_metrics(0.06)},
        "global_mean_velocity": {"metrics": _method_metrics(0.08)},
    }


def test_confirmation_gate_conjunction_passes_fixed_example() -> None:
    results, improvements = evaluate_confirmation_gates(
        _methods(), histogram_permutation_error=0.0, gates=_gates()
    )
    assert results["all_except_repeat"]
    assert improvements["median_error_improvement_over_hard_1nn_fraction"] == 0.5


def test_confirmation_gate_conjunction_detects_control_and_structure_failures() -> None:
    methods = deepcopy(_methods())
    methods["hard_1nn_hellinger"]["metrics"]["operator_output_rmse"][
        "median"
    ] = 0.04
    methods["query_kde_0p12"]["metrics"]["structure"][
        "minimum_jacobian_determinant"
    ] = 0.001
    results, _ = evaluate_confirmation_gates(
        methods, histogram_permutation_error=0.0, gates=_gates()
    )
    assert not results["improvement_over_hard"]
    assert not results["positive_jacobian"]
    assert not results["all_except_repeat"]
