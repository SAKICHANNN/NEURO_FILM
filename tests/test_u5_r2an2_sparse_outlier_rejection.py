from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2an2_two_stage_sparse_outlier_rejection import (
    CONFIG_SHA256,
    load_config,
    retained_support_metrics,
)
from src.roll2film.positive_film_sparse_rejection import (
    rank_sparse_residual_rejections,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2an2_two_stage_sparse_outlier_rejection_v1.json"


def test_frozen_contract_is_label_blind_and_synthetic_only() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert config["algorithm_may_read_true_outlier_rows"] is False
    assert config["two_stage_fit"]["rejected_row_fraction"] == 0.04
    assert config["two_stage_fit"]["rejected_row_count_rule"] == "floor"
    assert config["proxy_pair_allowed"] is False
    assert config["photographic_render_allowed"] is False


def test_residual_ranking_is_fixed_fraction_and_index_stable() -> None:
    scores = np.zeros(100, dtype=np.float64)
    scores[[50, 4, 90, 2, 70]] = 1.0
    rejected, retained = rank_sparse_residual_rejections(
        scores, rejected_row_fraction=0.04
    )
    assert np.array_equal(rejected, np.asarray([2, 4, 50, 70]))
    assert retained.sum() == 96
    assert np.all(~retained[rejected])
    assert not rejected.flags.writeable
    assert not retained.flags.writeable


def test_residual_ranking_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="invalid"):
        rank_sparse_residual_rejections(
            np.zeros((12, 1)), rejected_row_fraction=0.04
        )
    with pytest.raises(ValueError, match="invalid"):
        rank_sparse_residual_rejections(
            np.full(20, np.nan), rejected_row_fraction=0.04
        )
    with pytest.raises(ValueError, match="invalid"):
        rank_sparse_residual_rejections(
            np.zeros(20), rejected_row_fraction=0.0
        )


def test_retained_support_metrics_are_group_explicit() -> None:
    patch = np.repeat(np.arange(4), 6)
    illuminant = np.tile(np.repeat(np.arange(2), 3), 4)
    exposure = np.tile(np.arange(3), 8)
    retained = np.ones(24, dtype=bool)
    retained[[0, 23]] = False
    support = retained_support_metrics(
        retained, patch, illuminant, exposure
    )
    assert support == {
        "retained_row_count": 22,
        "rejected_row_count": 2,
        "distinct_base_patch_count": 4,
        "distinct_illuminant_count": 2,
        "distinct_exposure_count": 3,
        "minimum_retained_rows_per_base_patch": 5,
        "minimum_retained_rows_per_illuminant_exposure_cell": 3,
    }
    with pytest.raises(ValueError, match="align"):
        retained_support_metrics(
            retained[:-1], patch, illuminant, exposure
        )


def test_config_hash_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["two_stage_fit"]["rejected_row_fraction"] = 0.05
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(tampered, expected_sha256=CONFIG_SHA256)
