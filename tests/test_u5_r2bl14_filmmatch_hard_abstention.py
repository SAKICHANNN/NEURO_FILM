import numpy as np

from src.eval.filmmatch_hard_abstention import select_hard_policy, source_median_luma


def _rgb(value: float) -> np.ndarray:
    return np.full((4, 3), value, dtype=np.float64)


def test_source_median_luma_is_source_only_and_finite() -> None:
    source = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert source_median_luma(source) == np.median(source @ np.asarray([0.2126, 0.7152, 0.0722]))


def test_policy_chooses_high_support_without_blending() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    targets = [_rgb(0.5) for _ in scores]
    global_predictions = [_rgb(0.4) for _ in scores]
    adaptive_predictions = [_rgb(0.2), _rgb(0.2), _rgb(0.49), _rgb(0.49)]
    routing = {
        "threshold_quantiles": [0.5],
        "minimum_oof_improvement_over_global": 0.03,
        "minimum_oof_active_fraction": 0.15,
        "maximum_oof_active_fraction": 0.55,
        "minimum_oof_active_group_win_fraction": 2 / 3,
        "maximum_oof_p95_ratio_to_global": 1.0,
        "maximum_oof_worst_group_rmse_ratio_to_global": 1.05,
    }
    policy, rows = select_hard_policy(
        scores=scores,
        adaptive_predictions=adaptive_predictions,
        global_predictions=global_predictions,
        targets=targets,
        routing=routing,
    )
    assert policy is not None
    assert policy["active_groups"] == 2
    assert rows[0]["eligible"] is True


def test_policy_fails_closed_when_active_tail_is_harmful() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    targets = [_rgb(0.5) for _ in scores]
    global_predictions = [_rgb(0.4) for _ in scores]
    adaptive_predictions = [_rgb(0.4), _rgb(0.4), _rgb(0.0), _rgb(0.49)]
    routing = {
        "threshold_quantiles": [0.5],
        "minimum_oof_improvement_over_global": 0.03,
        "minimum_oof_active_fraction": 0.15,
        "maximum_oof_active_fraction": 0.55,
        "minimum_oof_active_group_win_fraction": 2 / 3,
        "maximum_oof_p95_ratio_to_global": 1.0,
        "maximum_oof_worst_group_rmse_ratio_to_global": 1.05,
    }
    policy, rows = select_hard_policy(
        scores=scores,
        adaptive_predictions=adaptive_predictions,
        global_predictions=global_predictions,
        targets=targets,
        routing=routing,
    )
    assert policy is None
    assert rows[0]["eligible"] is False
