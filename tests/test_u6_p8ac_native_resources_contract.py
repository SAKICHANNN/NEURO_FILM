from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.benchmark_u6_p8ac_native_resources import (
    _analytic_scene,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_p8ac_contract_and_scene_are_frozen() -> None:
    config = json.loads(
        (ROOT / "configs/u6_p8ac_native_resources_v1.json").read_text()
    )
    validate_contract(config)
    assert config["tile_rows"] == 32
    assert config["scenario"]["height"] * config["scenario"]["width"] == 12_000_000
    assert config["scenario"]["repeats"] == 2
    scene = _analytic_scene(7, 11)
    assert scene.shape == (7, 11, 3)
    assert scene.dtype == np.float64
    assert np.all(np.isfinite(scene))
    assert np.all((scene >= 0.0) & (scene <= 1.0))
