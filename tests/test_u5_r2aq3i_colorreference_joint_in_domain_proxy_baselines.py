from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2aq3i_colorreference_joint_in_domain_proxy_baselines import (
    CONFIG_SHA256,
    fold_masks,
    load_config,
    source_colour_folds,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq3i_colorreference_joint_in_domain_proxy_baselines_v1.json"
)


def test_contract_preserves_negative_evidence_and_claim_boundary() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    assert config["fit_allowed"] is True
    assert config["training_allowed"] is False
    assert config["render_allowed"] is False
    assert config["production_integration_allowed"] is False
    assert config["frozen_negative_evidence"][
        "aq2_whole_slide_extrapolation"
    ] == "closed and not reopened"


def test_source_fold_uses_exact_uint8_rgb_and_groups_duplicates() -> None:
    source = np.asarray(
        [[0, 0, 0], [28, 56, 85], [0, 0, 0], [255, 245, 245]],
        dtype=np.float64,
    ) / 255.0
    folds = source_colour_folds(source, salt="aq3t-v1")
    assert folds[0] == folds[2]
    with pytest.raises(ValueError, match="uint8-derived"):
        source_colour_folds(
            np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
            salt="aq3t-v1",
        )


def test_joint_masks_have_no_overlap_and_leave_buffer() -> None:
    table = {
        "test_set": np.asarray([1, 1, 2, 2], dtype=np.int64),
    }
    folds = np.asarray([0, 1, 0, 1], dtype=np.int64)
    development, confirmation = fold_masks(
        table,
        folds,
        held_set=1,
        held_fold=0,
    )
    np.testing.assert_array_equal(
        development, np.asarray([False, False, False, True])
    )
    np.testing.assert_array_equal(
        confirmation, np.asarray([True, False, False, False])
    )
    assert not np.any(development & confirmation)


def test_config_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["render_allowed"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(path, CONFIG_SHA256)
