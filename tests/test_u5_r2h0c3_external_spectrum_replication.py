from __future__ import annotations

import numpy as np
import pytest

from src.eval.external_spectrum_replication import (
    ExternalSpectrumError,
    policy_group_bootstrap,
    sample_group_from_member,
)


PATTERN = r"^splib07a_(.*)_(?:ASDFR|BECK|AVIRIS|NIC4)[A-Za-z]*_(?:AREF|RREF|TRAN|RTGC)$"


def test_sample_group_preserves_observed_sample_identity() -> None:
    member = (
        "ASCIIdata_splib07a/ChapterM_Minerals/"
        "splib07a_Actinolite_HS116.3B_ASDFRa_AREF.txt"
    )
    assert sample_group_from_member(member, PATTERN) == "Actinolite_HS116.3B"


def test_sample_group_fails_closed_on_unknown_name() -> None:
    with pytest.raises(ExternalSpectrumError):
        sample_group_from_member("root/not_a_record.txt", PATTERN)


def test_group_bootstrap_is_exact_and_group_weighted() -> None:
    groups = np.array(["a", "a", "b", "c", "d", "e"])
    smooth = np.full(6, 4.0)
    hard = np.full(6, 1.0)
    first = policy_group_bootstrap(groups, smooth, hard, 1e-12, 30, 19)
    second = policy_group_bootstrap(groups, smooth, hard, 1e-12, 30, 19)
    assert first == second
    assert first["win_rate"]["lower"] == 1.0
    assert first["median_relative_error_reduction"]["lower"] == 0.75

