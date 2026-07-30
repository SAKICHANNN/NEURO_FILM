from __future__ import annotations

import numpy as np

from src.eval.filmmatch_condition_oracle import _held_group_oracle


def test_held_group_oracle_excludes_complete_group() -> None:
    source = np.tile(
        np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), (12, 1)
    )
    target = source.copy()
    condition = np.asarray(["a"] * 12 + ["a"] * 12)
    groups = np.asarray(["one"] * 12 + ["two"] * 12)
    config = {
        "candidate": {
            "fit": {
                "curve_identity_mixture": 0.25,
                "matrix_identity_mixture": 0.25,
                "free_logit_bounds": [-8.0, 4.0],
                "restart_count": 1,
                "maximum_function_evaluations": 10,
                "function_tolerance": 1e-6,
                "parameter_tolerance": 1e-6,
                "gradient_tolerance": 1e-6,
                "loss": "linear",
                "loss_scale": 0.02,
                "seed": 1
            }
        }
    }
    rows = _held_group_oracle(
        source, target, condition, groups, config=config
    )
    assert len(rows) == 2
    assert all(row["same_condition_development_samples"] == 12 for row in rows)
