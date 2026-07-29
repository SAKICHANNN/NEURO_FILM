from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2aq3t_colorreference_calibration_suite_topology import (
    CONFIG_SHA256,
    colour_fold,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq3t_colorreference_calibration_suite_topology_v1.json"
)


def test_contract_preserves_aq2_and_is_source_only() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    assert config["fit_allowed"] is False
    assert config["target_fields_allowed"] is False
    assert config["aq2_reopened"] is False
    assert config["topology_contract"]["ordered_grid_prefix_rows"] == 1000
    assert config["topology_contract"]["ordered_hard_tail_rows"] == 152


def test_colour_fold_is_exact_duplicate_stable() -> None:
    values = np.array(
        [[0, 0, 0], [28, 56, 85], [0, 0, 0], [255, 245, 245]],
        dtype=np.uint8,
    )
    first = colour_fold(values, salt="aq3t-v1")
    second = colour_fold(values.copy(), salt="aq3t-v1")
    np.testing.assert_array_equal(first, second)
    assert first[0] == first[2]
    assert np.all((first >= 0) & (first < 5))


def test_colour_fold_rejects_non_rgb_or_out_of_range() -> None:
    with pytest.raises(ValueError, match="uint8-compatible"):
        colour_fold(np.zeros((2, 4), dtype=np.uint8), salt="aq3t-v1")
    with pytest.raises(ValueError, match="uint8-compatible"):
        colour_fold(np.array([[0, 0, 256]]), salt="aq3t-v1")


def test_config_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["aq2_reopened"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(path, CONFIG_SHA256)
