import json
from pathlib import Path

import numpy as np
import pytest

from src.roll2film.factorized_monotone_bernstein import (
    fit_factorized_monotone_bernstein,
)


def test_ax13_freezes_only_bounded_degree5_strength_candidates() -> None:
    config = json.loads(
        Path(
            "configs/u5_r2ax13_filmmatch_degree5_strength_cap_v1.json"
        ).read_text(encoding="utf-8")
    )
    variants = config["candidate"]["variants"]
    assert [row["residual_strength_cap"] for row in variants] == [
        0.7,
        0.65,
        0.6,
    ]
    assert {row["residual_degree"] for row in variants} == {5}
    assert (
        config["gates"]["automatic_safety_threshold_relaxation_allowed"]
        is False
    )
    assert config["gates"]["capacity_expansion_allowed"] is False


def test_factorized_fit_rejects_invalid_strength_cap() -> None:
    source = np.asarray([[0.2, 0.3, 0.4], [0.6, 0.5, 0.4]])
    with pytest.raises(ValueError, match="residual_strength_cap"):
        fit_factorized_monotone_bernstein(
            source,
            source,
            segment_count=2,
            curve_learned_mixture=0.5,
            matrix_identity_mixture=0.5,
            free_logit_bounds=(-2.0, 2.0),
            restart_count=1,
            maximum_function_evaluations=2,
            function_tolerance=1e-6,
            parameter_tolerance=1e-6,
            gradient_tolerance=1e-6,
            loss="linear",
            loss_scale=0.1,
            seed=1,
            residual_degree=2,
            residual_identity_ridge=1.0,
            jacobian_floor=1e-4,
            safety_grid_size=3,
            strength_steps=2,
            maximum_residual_iterations=2,
            residual_strength_cap=0.0,
        )
