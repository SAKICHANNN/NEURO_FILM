from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from src.eval.filmmatch_strict_interior_fresh_confirmation import (
    AO6_ARM,
    CANDIDATE_ARM,
    _canonical_sha256,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_v1.json"


def test_bl8_arm_identities_are_fixed_and_distinct() -> None:
    assert AO6_ARM == "fixed_ao6_colour_only_t15_c35"
    assert CANDIDATE_ARM == "fixed_bl5_strict_interior_sigmoid"
    assert AO6_ARM != CANDIDATE_ARM


def test_bl8_canonical_identity_is_order_independent() -> None:
    left = {"b": [np.float32(0.25).item()], "a": 1}
    right = {"a": 1, "b": [0.25]}
    assert _canonical_sha256(left) == _canonical_sha256(right)


def test_bl8_real_contract_binds_fresh_population_and_fixed_arms() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["eligible_ids"]) == 17
    assert len({validated["source_rows"][key]["make"] for key in validated["eligible_ids"]}) == 17
    assert config["automatic_gate"]["expected_outputs"] == 34
    assert config["blind_protocol"]["minimum_candidate_aggregate_choices"] == 31
    assert config["operator_fitting_allowed"] is False
