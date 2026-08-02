from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.fivek_bilateral_gain_capacity import _held_mask
from src.eval.fivek_semantic_gated_curves import (
    _apply_curves,
    _curve_features,
    _fit_curves,
    _zsigmoid,
    evaluate_semantic_gated_curves,
)
from src.eval.fivek_semantic_gated_curves_run import validate_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bv0_fivek_semantic_gated_curves_v1.json"


def test_bv0_contract_is_source_only_explicit_and_development_only() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    assert config["semantic_gate"]["training_or_finetuning_allowed"] is False
    assert config["explicit_operator"]["learned_final_rgb_allowed"] is False
    assert config["explicit_operator"]["hard_output_clipping_allowed"] is False
    assert config["explicit_operator"]["parameter_count"] == 63
    assert config["population"]["confirmation_rows_allowed"] is False


def test_curve_basis_has_exact_zero_endpoint_residuals() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    knots = config["explicit_operator"]["knot_positions"]
    source = np.asarray([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]])
    features = _curve_features(source, knots)
    assert features.shape == (3, 21)
    assert np.array_equal(features[0], np.zeros(21))
    assert np.array_equal(features[1], np.zeros(21))
    assert np.sum(features[2]) == 3.0


def test_analytical_curve_dose_is_bounded_without_clipping() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = config["explicit_operator"]
    source = np.full((9, 11, 3), 0.5)
    gate = np.linspace(0.1, 0.9, 99).reshape(9, 11)
    coefficients = np.full((21, 3), 0.75)
    output, dose = _apply_curves(source, gate, coefficients, spec)
    assert np.all(np.isfinite(output))
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.min(dose) < 1.0


def test_semantic_gate_beats_controls_on_synthetic_local_effect() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["population"]["source_count_exact"] = 2
    gates = config["evaluation"]["automatic_gates"]
    gates["source_count_exact"] = 2
    gates["minimum_semantic_gate_spatial_std"] = 0.01
    rows = []
    y, x = np.mgrid[0:48, 0:48]
    for index in range(2):
        source = np.stack(
            [
                0.1 + 0.75 * x / 47.0,
                0.12 + 0.7 * y / 47.0,
                0.18 + 0.55 * (x + y) / 94.0,
            ],
            axis=-1,
        )
        semantic = _zsigmoid(-((x - 14.0) ** 2 + (y - 18.0) ** 2) / 90.0)
        truth_coefficients = np.zeros((21, 3))
        truth_coefficients[3, 0] = 0.20
        truth_coefficients[10, 1] = -0.16
        truth_coefficients[17, 2] = 0.12
        target, _ = _apply_curves(
            source, semantic, truth_coefficients, config["explicit_operator"]
        )
        rows.append(
            {
                "pair_id": f"synthetic-{index}",
                "group": f"group-{index}",
                "target_variant": "aligned_expert",
                "source": source,
                "target": target,
                "semantic_gate": semantic,
            }
        )
    result = evaluate_semantic_gated_curves(rows, config)
    assert result["metrics"]["mean_improvement_over_global_gate"] > 0.0
    assert result["metrics"]["mean_improvement_over_shifted_semantic_gate"] > 0.0
    assert result["metrics"]["maximum_out_of_cube_fraction"] == 0.0


def test_fit_mask_cannot_read_held_target_pixels() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = config["explicit_operator"]
    rng = np.random.default_rng(12)
    source = rng.uniform(0.05, 0.95, size=(24, 24, 3))
    gate = _zsigmoid(rng.normal(size=(24, 24)))
    held = _held_mask(source.shape[:2], config["held_block_protocol"], 0)
    first = source.copy()
    second = source.copy()
    first[~held] += 0.02
    second[~held] += 0.02
    second[held] = 1.0 - second[held]
    a = _fit_curves(source, first, gate, ~held, spec)
    b = _fit_curves(source, second, gate, ~held, spec)
    assert np.array_equal(a, b)
