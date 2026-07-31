from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.synthetic_hard_mode_identifiability import (
    SyntheticHardModeError,
    canonical_json_bytes,
    cosine_distance,
    evaluate,
    load_config,
    run,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk19_synthetic_hard_mode_identifiability_v1.json"


def test_frozen_contract_binds_closed_bk18() -> None:
    config = load_config(ROOT, CONFIG)
    assert config["training_allowed"] is False
    assert config["operator_fitting_allowed"] is False
    assert config["production_integration_allowed"] is False


def test_wrong_parent_decision_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["parent"]["required_decision"] = "not-the-frozen-decision"
    candidate = tmp_path / "config.json"
    candidate.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(SyntheticHardModeError, match="parent decision mismatch"):
        load_config(ROOT, candidate)


def test_cosine_distance_identity_and_orthogonal() -> None:
    assert cosine_distance(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])) == 0.0
    assert cosine_distance(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == 1.0


def test_known_modes_and_strength_negative_control_pass() -> None:
    config = load_config(ROOT, CONFIG)
    report = evaluate(config, config_sha256=sha256_file(CONFIG))
    assert report["automatic_pass"] is True
    assert report["true_mode"]["leave_one_group_out_accuracy"] >= 0.9
    assert (
        report["true_mode"]["k2_absolute_directional_error_improvement"] >= 0.5
    )
    assert (
        report["strength_path_negative_control"][
            "minimum_same_group_cross_strength_cosine"
        ]
        >= 0.995
    )
    assert (
        report["strength_path_negative_control"][
            "k2_absolute_directional_error_improvement"
        ]
        <= 0.05
    )


def test_two_runs_are_byte_exact(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    run(ROOT, CONFIG, first)
    run(ROOT, CONFIG, second)
    assert first.read_bytes() == second.read_bytes()
