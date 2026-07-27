from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.evaluate_statlut_lite import (
    _fit_ridge,
    _predict,
    _rows,
    build_samples,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_statlut_lite_v1.json"


def test_statlut_lite_contract_reserves_last_stress_half() -> None:
    config, _, manifest = load_contract(CONFIG)
    rows = _rows(config, manifest, "stress")
    assert len(rows) == 48
    assert {
        int(row["within_split_index"]) for row in rows
    } == set(range(16, 32))


def test_statlut_lite_sample_has_coupled_feature_and_parameter_target() -> None:
    config, parent, manifest = load_contract(CONFIG)
    samples = build_samples(
        config,
        parent,
        _rows(config, manifest, "fit")[:1],
        "fit",
    )
    assert len(samples) == 2
    assert samples[0]["feature"].shape == (480,)
    assert samples[0]["truth_parameters"].shape == (9,)
    assert samples[0]["query"].shape == (1024, 3)
    assert samples[0]["target"].shape == (1024, 3)


def test_statlut_lite_ridge_repeat_and_projection_are_deterministic() -> None:
    rng = np.random.default_rng(2026072706)
    x = rng.normal(size=(32, 20))
    y = rng.uniform(size=(32, 9))
    first = _fit_ridge(x, y, 1.0)
    second = _fit_ridge(x, y, 1.0)
    prediction_a = _predict(first, x[:4])
    prediction_b = _predict(second, x[:4])
    assert np.array_equal(prediction_a, prediction_b)
    assert prediction_a.shape == (4, 9)
    assert np.all(prediction_a[:, :6] >= 0.0)
    assert np.all(np.diff(prediction_a[:, 6:9], axis=1) >= 0.08 - 1e-12)
