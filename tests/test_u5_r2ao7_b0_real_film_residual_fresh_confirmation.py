from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.b0_real_film_residual_fresh_confirmation import (
    FreshResidualConfirmationError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (
        ROOT
        / "configs/u5_r2ao7_b0_real_film_residual_fresh_confirmation_v1.json"
    ).read_text(encoding="utf-8")
)


def test_frozen_contract_validates_exact_fresh_population() -> None:
    validated = validate_contract(ROOT, CONFIG)
    assert validated["eligible_ids"] == CONFIG["confirmation_population"][
        "eligible_ids"
    ]
    assert len(validated["source_rows"]) == 16
    assert len({row["make"] for row in validated["source_rows"].values()}) == 9


def test_contract_rejects_development_threshold_drift() -> None:
    changed = copy.deepcopy(CONFIG)
    changed["metrics"]["minimum_confirmation_median_style_delta_e76"] += 0.01
    with pytest.raises(FreshResidualConfirmationError, match="threshold"):
        validate_contract(ROOT, changed)


def test_contract_rejects_post_render_population_change() -> None:
    changed = copy.deepcopy(CONFIG)
    changed["confirmation_population"]["eligible_ids"] = changed[
        "confirmation_population"
    ]["eligible_ids"][:-1]
    changed["confirmation_population"]["expected_rows"] = 15
    with pytest.raises(FreshResidualConfirmationError, match="population"):
        validate_contract(ROOT, changed)


def test_contract_rejects_operator_strength_rescue() -> None:
    changed = copy.deepcopy(CONFIG)
    changed["fixed_candidate"]["chroma_strength"] = 0.36
    with pytest.raises(FreshResidualConfirmationError, match="candidate"):
        validate_contract(ROOT, changed)
