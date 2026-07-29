from __future__ import annotations

import numpy as np
import pytest

from src.eval.apollo16_bw_step_chart_feasibility import (
    Apollo16BWWedgeFeasibilityError,
    supported_level_groups,
)


def test_supported_level_groups_keep_six_separated_plateaus() -> None:
    profile = np.repeat(
        np.asarray([1000, 8000, 16000, 26000, 40000, 60000], dtype=np.float64),
        40,
    )
    groups = supported_level_groups(
        profile,
        bin_width_u16=1024,
        minimum_support_fraction=0.02,
    )
    assert len(groups) == 6
    assert all(row["support_rows"] == 40 for row in groups)


def test_adjacent_supported_bins_merge_into_one_level() -> None:
    profile = np.concatenate(
        (
            np.full(80, 10),
            np.full(40, 60 * 1024 + 100),
            np.full(40, 61 * 1024 + 100),
        )
    )
    groups = supported_level_groups(
        profile,
        bin_width_u16=1024,
        minimum_support_fraction=0.02,
    )
    assert len(groups) == 2
    assert groups[1]["first_bin"] == 60
    assert groups[1]["last_bin"] == 61
    assert groups[1]["support_rows"] == 80


@pytest.mark.parametrize(
    "profile,bin_width,support",
    [
        (np.asarray([]), 1024, 0.02),
        (np.asarray([np.nan]), 1024, 0.02),
        (np.asarray([70000]), 1024, 0.02),
        (np.asarray([1]), 0, 0.02),
        (np.asarray([1]), 1024, 0.0),
    ],
)
def test_invalid_profiles_fail_closed(
    profile: np.ndarray, bin_width: int, support: float
) -> None:
    with pytest.raises(Apollo16BWWedgeFeasibilityError):
        supported_level_groups(
            profile,
            bin_width_u16=bin_width,
            minimum_support_fraction=support,
        )
