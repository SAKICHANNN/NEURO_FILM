from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.monotone_source_tone_fraction_transport import (
    select_monotone_source_tone_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb38_exact_lab_y_monotone_tone_development_v1.json"
DECISION = ROOT / "configs/u5_r2cb38_exact_lab_y_monotone_tone_development_decision_v1.json"


def test_cb38_contract_freezes_exact_lab_y_on_distinct_population() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB38"
    assert contract["population"]["source_count_exact"] == 11
    assert contract["population"]["exact_decoded_sha_overlap_with_cb27_through_cb37"] == 0
    assert contract["operator"]["luminance_weights"] == [0.212671, 0.71516, 0.072169]


def test_cb38_weights_reproduce_legacy_lab_lstar_order() -> None:
    rng = np.random.default_rng(20260812)
    source = rng.uniform(0.01, 0.99, size=(29, 31, 3)).astype(np.float32)
    y = np.sum(source.astype(np.float64) * LEGACY_LAB_Y_WEIGHTS, axis=-1)
    lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    assert np.array_equal(np.argsort(y.reshape(-1)), np.argsort(lstar.reshape(-1)))


def test_cb38_selector_uses_exact_lab_y_without_inversion() -> None:
    x = np.linspace(0.03, 0.91, 53, dtype=np.float32)
    source = x[None, :, None]
    source = np.repeat(source, 3, axis=2)
    base = np.asarray(np.sqrt(source), dtype=np.float32)
    target = base.copy()
    target[..., 0] += np.float32(0.05)
    target[..., 2] -= np.float32(0.03)
    _, _, luma_error, facts = select_monotone_source_tone_candidate(
        source,
        base,
        target,
        weights=LEGACY_LAB_Y_WEIGHTS,
        boundary_epsilon=1.0 / 65535.0,
        tone_knot_count=257,
        dose_grid=[1.0, 0.5, 0.0],
        maximum_gradient_ratio=10.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
    )
    assert float(np.max(np.abs(luma_error))) < 1e-6
    assert facts["selected_lstar_inversion_fraction"] == 0.0


def test_cb38_decision_requires_direct_lstar_coordinate() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["failure"]["minimum_gradient_ratio"] < 1.35
    assert decision["failure"]["minimum_lstar_inversion_fraction"] > 0.0
    assert decision["decision"] == (
        "close_exact_weight_proxy_open_direct_lstar_successor"
    )
    assert decision["visual_review_status"] == "forbidden"
