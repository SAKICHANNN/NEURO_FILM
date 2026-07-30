from __future__ import annotations

import numpy as np

from src.eval.filmmatch_code_domain_capacity import (
    MODEL_NAMES,
    prediction_metrics,
    select_development_candidate,
)


def test_prediction_metrics_report_sample_level_oog() -> None:
    target = np.asarray([[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]])
    prediction = target.copy()
    prediction[1, 2] = 1.1
    metrics = prediction_metrics(prediction, target)
    assert metrics["out_of_cube_sample_fraction"] == 0.5
    assert metrics["rgb_rmse"] > 0.0


def test_selection_excludes_unsafe_and_prefers_simpler_near_best() -> None:
    reflective = {}
    hue = {}
    structural = {}
    for index, name in enumerate(MODEL_NAMES):
        reflective[name] = {
            "mean_rgb_rmse": 1.0 - index * 0.05,
            "worst_out_of_cube_sample_fraction": 0.0,
        }
        hue[name] = {
            "mean_rgb_rmse": 1.0 - index * 0.05,
            "worst_out_of_cube_sample_fraction": 0.0,
        }
        structural[name] = {"safe": True}
    structural["emulating_emulsion_equation_30p"]["safe"] = False
    reflective["bounded_positive_one_matrix"]["mean_rgb_rmse"] = 0.5
    hue["bounded_positive_one_matrix"]["mean_rgb_rmse"] = 0.5
    reflective["bounded_positive_two_matrix"]["mean_rgb_rmse"] = 0.495
    hue["bounded_positive_two_matrix"]["mean_rgb_rmse"] = 0.495
    result = select_development_candidate(reflective, hue, structural)
    assert result["selected"] == "bounded_positive_one_matrix"
    assert "emulating_emulsion_equation_30p" not in result["eligible"]
