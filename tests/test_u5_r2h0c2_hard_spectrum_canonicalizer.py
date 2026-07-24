from __future__ import annotations

import numpy as np

from src.eval.cave_conditional_variability import RepresentativePopulation
from src.eval.hard_spectrum_canonicalizer import (
    evaluate_threshold,
    nearest_other_scene,
    policy_scene_bootstrap,
)


def _population() -> RepresentativePopulation:
    lab = np.array(
        [[50.0, 0.0, 0.0], [50.1, 0.0, 0.0], [60.0, 0.0, 0.0], [60.1, 0.0, 0.0]]
    )
    return RepresentativePopulation(
        spectra=np.zeros((4, 3)),
        xyz=np.zeros((4, 3)),
        lab=lab,
        scene=np.array(["a", "b", "a", "c"]),
        cell_row=np.zeros(4, dtype=np.int16),
        cell_column=np.arange(4, dtype=np.int16),
    )


def test_nearest_other_scene_never_returns_query_scene() -> None:
    population = _population()
    nearest, distance = nearest_other_scene(population)
    assert np.all(population.scene != population.scene[nearest])
    assert np.all(distance >= 0)
    assert nearest[0] == 1


def test_scene_bootstrap_is_deterministic() -> None:
    scenes = np.array(["a", "a", "b", "b", "c", "c"])
    smooth = np.array([3.0, 4.0, 3.0, 4.0, 3.0, 4.0])
    hard = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    first = policy_scene_bootstrap(scenes, smooth, hard, 1e-12, 20, 7)
    second = policy_scene_bootstrap(scenes, smooth, hard, 1e-12, 20, 7)
    assert first == second
    assert first["win_rate"]["lower"] == 1.0


def test_threshold_uses_smooth_fallback_and_all_gates() -> None:
    population = RepresentativePopulation(
        spectra=np.zeros((6, 3)),
        xyz=np.zeros((6, 3)),
        lab=np.zeros((6, 3)),
        scene=np.array(["a", "b", "c", "a", "b", "c"]),
        cell_row=np.zeros(6, dtype=np.int16),
        cell_column=np.arange(6, dtype=np.int16),
    )
    config = {
        "seed": 3,
        "tie_tolerance_delta_e76": 1e-12,
        "gates": {
            "selected_queries_min": 3,
            "coverage_min": 0.5,
            "selected_scenes_min": 3,
            "max_selected_scene_share": 0.5,
            "selected_win_rate_min": 0.6,
            "selected_win_rate_bootstrap_95_lcb_min": 0.5,
            "selected_median_relative_error_reduction_min": 0.2,
            "selected_median_relative_error_reduction_bootstrap_95_lcb_min": 0.0,
            "selected_hard_error_delta_e76_median_max": 5.0,
            "selected_hard_error_delta_e76_p95_max": 12.0,
            "fallback_policy_p95_vs_smooth_max_increase": 0.0,
            "scene_bootstrap_repeats": 20,
        },
    }
    distance = np.array([0.1, 0.2, 0.3, 2.0, 2.0, 2.0])
    smooth = np.full(6, 4.0)
    hard = np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
    report, policy_error = evaluate_threshold(
        1.0, population, distance, smooth, hard, config
    )
    assert report["eligible"]
    assert np.array_equal(policy_error, np.array([1.0, 1.0, 1.0, 4.0, 4.0, 4.0]))
