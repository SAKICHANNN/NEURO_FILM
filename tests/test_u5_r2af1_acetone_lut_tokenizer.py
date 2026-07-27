from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.acetone_lut_tokenizer import (
    analytic_lut,
    build_population,
    delta_e_summary,
    metric_negative_control,
)
from src.eval.spectral_film_lut_bank import (
    local_jacobian_metrics,
    synthetic_cube,
)


CONFIG = Path("configs/u5_r2af1_acetone_tokenizer_topology_v1.json")


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_analytic_population_is_safe() -> None:
    config = load_config()
    population = build_population(config)
    assert list(population) == [
        row["id"] for row in config["population"]["records"]
    ]
    assert len(population) == 8
    for value in population.values():
        assert value.shape == (32, 32, 32, 3)
        assert value.dtype == np.float32
        assert np.all(np.isfinite(value))
        assert float(np.min(value)) >= 0.0
        assert float(np.max(value)) <= 1.0
        metrics = local_jacobian_metrics(value)
        assert metrics["negative_jacobian_fraction"] == 0.0
        assert metrics["minimum_jacobian_determinant"] >= 0.25


def test_identity_record_is_exact_float32_cube() -> None:
    population = build_population(load_config())
    assert np.array_equal(
        population["identity"], synthetic_cube(32).astype(np.float32)
    )


def test_metric_negative_control_reverses_orientation() -> None:
    assert metric_negative_control(32) == 1.0


def test_delta_e_identity_is_zero() -> None:
    cube = synthetic_cube(9)
    summary = delta_e_summary(cube, cube.copy())
    assert summary["median_delta_e76"] == 0.0
    assert summary["p95_delta_e76"] == 0.0


def test_analytic_lut_rejects_invalid_parameters() -> None:
    cube = synthetic_cube(5)
    try:
        analytic_lut(cube, [0.0], np.eye(3).tolist())
    except ValueError as exc:
        assert "three curve terms" in str(exc)
    else:
        raise AssertionError("invalid analytic curve must fail closed")
