from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2an1_robust_paired_positive_film_recovery import (
    CONFIG_SHA256,
    load_config,
    perturb_development_pairs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2an1_robust_paired_positive_film_recovery_v1.json"


def test_frozen_contract_and_perturbations_are_exact() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    source = np.linspace(0.0, 1.0, 120 * 3).reshape(120, 3)
    target = source**0.8
    first = perturb_development_pairs(
        source,
        target,
        config["scenarios"]["sparse_correspondence_outliers"],
        seed=config["perturbation_seed"],
    )
    second = perturb_development_pairs(
        source,
        target,
        config["scenarios"]["sparse_correspondence_outliers"],
        seed=config["perturbation_seed"],
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    assert first[2].size == 4
    assert np.all(np.diff(first[2]) > 0)
    assert np.min(first[0]) >= 0.0 and np.max(first[0]) <= 1.0
    assert np.min(first[1]) >= 0.0 and np.max(first[1]) <= 1.0


def test_invalid_perturbation_pairs_fail_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    scenario = config["scenarios"]["dual_patch_mean_noise"]
    with pytest.raises(ValueError, match="paired"):
        perturb_development_pairs(
            np.zeros((12, 3)), np.zeros((11, 3)), scenario, seed=1
        )
    with pytest.raises(ValueError, match="paired"):
        perturb_development_pairs(
            np.full((12, 3), np.nan), np.zeros((12, 3)), scenario, seed=1
        )
