from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2aq3p_colorreference_bernstein_bundle_properties import (
    CONFIG_SHA256,
    load_config,
    local_axis_distances,
    regular_grid,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq3p_colorreference_bernstein_bundle_properties_v1.json"
)


def test_contract_preserves_recorder_semantics_and_no_render() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    bundle = config["bundle_contract"]
    assert bundle["model_id"] == "bernstein_d3"
    assert "recorder-device" in bundle["input_semantics"]
    assert bundle["post_fit_clipping"] is False
    assert config["render_allowed"] is False
    assert config["production_integration_allowed"] is False


def test_regular_grid_and_local_distances() -> None:
    grid = regular_grid(3)
    assert grid.shape == (27, 3)
    assert float(np.min(grid)) == 0.0
    assert float(np.max(grid)) == 1.0
    distance = local_axis_distances(grid, size=3)
    assert distance.shape == (54,)
    np.testing.assert_allclose(distance, 0.5)


def test_invalid_grid_shape_fails_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        local_axis_distances(np.zeros((8, 2)), size=2)


def test_config_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bundle_contract"]["input_semantics"] = "sRGB"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(path, CONFIG_SHA256)
