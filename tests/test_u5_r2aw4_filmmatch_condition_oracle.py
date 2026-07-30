from __future__ import annotations

import numpy as np

from src.eval.filmmatch_condition_oracle import _aggregate


def test_oracle_aggregate_counts_wins_and_median() -> None:
    rows = []
    for improvement in (0.2, 0.1, -0.1):
        rows.append(
            {
                "expert_improvement_over_global": improvement,
                "metrics": {
                    "global": {"rgb_rmse": 1.0},
                    "correct_condition_expert": {
                        "rgb_rmse": 1.0 - improvement
                    },
                },
            }
        )
    aggregate = _aggregate(rows)
    assert aggregate["expert_wins"] == 2
    assert np.isclose(aggregate["expert_win_fraction"], 2.0 / 3.0)
    assert np.isclose(aggregate["median_improvement_over_global"], 0.1)
