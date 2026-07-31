from __future__ import annotations

import numpy as np

from src.eval.filmmatch_strict_interior_fresh_confirmation import (
    AO6_ARM,
    CANDIDATE_ARM,
    _canonical_sha256,
)


def test_bl8_arm_identities_are_fixed_and_distinct() -> None:
    assert AO6_ARM == "fixed_ao6_colour_only_t15_c35"
    assert CANDIDATE_ARM == "fixed_bl5_strict_interior_sigmoid"
    assert AO6_ARM != CANDIDATE_ARM


def test_bl8_canonical_identity_is_order_independent() -> None:
    left = {"b": [np.float32(0.25).item()], "a": 1}
    right = {"a": 1, "b": [0.25]}
    assert _canonical_sha256(left) == _canonical_sha256(right)
