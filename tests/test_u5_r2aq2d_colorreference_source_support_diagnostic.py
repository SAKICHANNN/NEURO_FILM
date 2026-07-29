from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2aq2d_colorreference_source_support_diagnostic import (
    CONFIG_SHA256,
    load_config,
    source_support_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq2d_colorreference_source_support_diagnostic_v1.json"
)


def test_contract_is_source_only_and_non_promoting() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    assert config["fit_allowed"] is False
    assert config["target_fields_allowed"] is False
    assert config["operator_promotion_allowed"] is False
    assert config["capacity_rescue_allowed"] is False


def test_support_metrics_distinguish_inside_and_outside() -> None:
    rng = np.random.default_rng(20260729)
    development = rng.uniform(0.0, 1.0, size=(1152, 3))
    held_inside = rng.uniform(0.2, 0.8, size=(288, 3))
    inside = source_support_metrics(
        held_inside, development, hull_tolerance=1e-10
    )
    assert inside["held_outside_development_convex_hull_fraction"] == 0.0
    held_outside = held_inside.copy()
    held_outside[:144, 0] = 2.0
    outside = source_support_metrics(
        held_outside, development, hull_tolerance=1e-10
    )
    assert outside["held_outside_development_convex_hull_fraction"] >= 0.5
    assert outside["held_nearest_distance_p95"] > inside[
        "held_nearest_distance_p95"
    ]


def test_exact_overlap_is_reported() -> None:
    rng = np.random.default_rng(42)
    development = rng.uniform(0.0, 1.0, size=(1152, 3))
    held = development[:288].copy()
    metrics = source_support_metrics(
        held, development, hull_tolerance=1e-10
    )
    assert metrics["held_exact_rgb_overlap_fraction"] == 1.0
    assert metrics["held_nearest_distance_maximum"] == 0.0


def test_config_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["operator_promotion_allowed"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(path, CONFIG_SHA256)
