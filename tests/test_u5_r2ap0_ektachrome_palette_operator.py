from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ap0_ektachrome_palette_operator import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.ektachrome_palette_operator import (
    complete_row_folds,
    evaluate_ektachrome_palette_operator,
    load_ektachrome_pairs,
    operator_direction_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ap0_ektachrome_palette_operator_v1.json"


def test_contract_identity_lineage_and_claims() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    assert config["source"]["expected_pair_count"] == 56
    assert config["source"]["row_cell_counts"] == [10, 10, 10, 10, 10, 6]
    assert config["operator_fitting_allowed"] is True
    assert config["image_rendering_allowed"] is False
    assert config["stock_response_claim_allowed"] is False
    assert config["calibrated_reference_claim_allowed"] is False


def test_contract_hash_fails_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG, expected_sha256="0" * 64)


def test_exact_pairs_and_complete_row_partition() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    source, target = load_ektachrome_pairs(
        ROOT / config["source"]["paired_palettes"], config
    )
    assert source.shape == target.shape == (56, 3)
    assert np.all((source >= 0.0) & (source <= 1.0))
    assert np.all((target >= 0.0) & (target <= 1.0))
    folds = complete_row_folds(
        config["source"]["row_cell_counts"], total=len(source)
    )
    assert [len(fold) for fold in folds] == [10, 10, 10, 10, 10, 6]
    assert np.array_equal(np.concatenate(folds), np.arange(56))


def test_row_partition_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="do not partition"):
        complete_row_folds([10, 10], total=21)


def test_direction_metric_distinguishes_scale_from_new_direction() -> None:
    control = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    scaled = 2.0 * control
    orthogonal_component = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    same = operator_direction_metrics(scaled, control)
    different = operator_direction_metrics(
        scaled + orthogonal_component, control
    )
    assert same["absolute_cosine"] == pytest.approx(1.0)
    assert same["residual_fraction_after_best_scalar_alignment"] == pytest.approx(
        0.0
    )
    assert different["absolute_cosine"] < 1.0
    assert different["residual_fraction_after_best_scalar_alignment"] > 0.0


def test_frozen_complete_row_evaluation_passes_without_claim_inflation() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    source, target = load_ektachrome_pairs(
        ROOT / config["source"]["paired_palettes"], config
    )
    direction = config["velvia_direction_control"]
    operator_config = json.loads(
        (ROOT / direction["operator_config"]).read_bytes()
    )
    result = evaluate_ektachrome_palette_operator(
        source,
        target,
        config,
        velvia_operator_payload=operator_config["witnesses"][
            direction["operator_witness_id"]
        ],
    )
    assert result["automatic_pass"] is True
    assert result["selected_model"] == "one_matrix"
    assert len(result["folds"]) == 6
    assert result["fold_wins_over_identity"] == 5
    assert result["fold_wins_over_full_affine"] == 6
    assert result["cross_validated_metrics"]["one_matrix"][
        "raw_out_of_cube_fraction"
    ] == 0.0
