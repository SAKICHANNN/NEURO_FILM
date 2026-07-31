from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.factorized_ao6_sixth_fresh_confirmation import (
    ARMS,
    render_fixed_arms,
    validate_contract,
)
from src.eval.log_chroma_fresh_comparison import LogChromaFreshComparisonError
from src.film_physics.profile_consumer import compile_standalone_profile_artifact


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2bk18_factorized_ao6_sixth_fresh_confirmation_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_bk18_contract_binds_fixed_arms_and_sixth_source() -> None:
    config = _config()
    validated = validate_contract(ROOT, config)
    assert tuple(row["arm_id"] for row in config["fixed_arms"]) == ARMS
    assert len(validated["source_rows"]) == 12
    assert len({row["make"] for row in validated["source_rows"]}) == 12
    assert config["automatic_gate"]["expected_outputs"] == 48
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["selector_training_allowed"]
    assert not config["rendering"]["strength_retuning_allowed"]
    assert not config["rendering"]["routing_allowed"]
    assert not config["rendering"]["dense_blending_allowed"]


def test_bk18_render_is_finite_bounded_repeat_exact_and_not_bk7() -> None:
    validated = validate_contract(ROOT, _config())
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=validated["ao6_config"]
    )
    source = np.random.default_rng(20260731).uniform(
        0.02, 0.98, size=(9, 13, 3)
    ).astype(np.float32)
    first = render_fixed_arms(source, validated=validated, ao6_artifact=artifact)
    second = render_fixed_arms(source, validated=validated, ao6_artifact=artifact)
    assert set(first) == set(ARMS)
    for arm in ARMS:
        assert first[arm].shape == source.shape
        assert np.all(np.isfinite(first[arm]))
        assert np.all((first[arm] >= 0.0) & (first[arm] <= 1.0))
        assert np.array_equal(first[arm], second[arm])
    assert not np.array_equal(first[ARMS[0]], first[ARMS[1]])


def test_bk18_rejects_contract_and_source_identity_drift() -> None:
    config = _config()
    config["rendering"]["strength_retuning_allowed"] = True
    with pytest.raises(LogChromaFreshComparisonError):
        validate_contract(ROOT, config)
    config = _config()
    config["source_preflight"]["decision_sha256"] = "0" * 64
    with pytest.raises(LogChromaFreshComparisonError):
        validate_contract(ROOT, config)
