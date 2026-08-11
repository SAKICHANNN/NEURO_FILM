from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.monotone_source_tone_fraction_transport import (
    monotone_quantile_luminance_map,
    select_monotone_source_tone_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb37_monotone_source_tone_fraction_development_v1.json"


def test_cb37_contract_freezes_distinct_population_and_monotone_tone() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB37"
    assert contract["population"]["source_count_exact"] == 10
    assert contract["population"]["exact_decoded_sha_overlap_with_cb27_cb28_cb34_cb35_cb36"] == 0
    assert contract["operator"]["tone_knot_count"] == 257
    assert contract["automatic_gates"]["maximum_adjacent_lstar_gradient_sign_inversion_fraction"] == 0.0


def test_cb37_quantile_tone_is_monotone_and_matches_target_range() -> None:
    source = np.asarray([[0.1, 0.1, 0.2], [0.3, 0.7, 0.9]], dtype=np.float64)
    target = np.asarray([[0.02, 0.04, 0.08], [0.4, 0.6, 0.95]], dtype=np.float64)
    mapped, facts = monotone_quantile_luminance_map(source, target, knot_count=17)
    order = np.argsort(source.reshape(-1), kind="stable")
    assert np.all(np.diff(mapped.reshape(-1)[order]) >= 0.0)
    assert float(np.min(target)) <= facts["tone_minimum"] <= facts["tone_maximum"]
    assert facts["tone_maximum"] == float(np.max(target))


def test_cb37_selector_keeps_monotone_tone_with_chroma() -> None:
    x = np.linspace(0.05, 0.85, 41, dtype=np.float32)
    source = np.repeat(x[None, :, None], 13, axis=0)
    source = np.repeat(source, 3, axis=2)
    safe_base = np.asarray(np.sqrt(source), dtype=np.float32)
    target = safe_base.copy()
    target[..., 0] += np.float32(0.04)
    target[..., 2] -= np.float32(0.02)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    candidate, scale, luma_error, facts = select_monotone_source_tone_candidate(
        source,
        safe_base,
        target,
        weights=weights,
        boundary_epsilon=1.0 / 65535.0,
        tone_knot_count=257,
        dose_grid=[1.0, 0.5, 0.0],
        maximum_gradient_ratio=10.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
    )
    assert candidate.shape == source.shape
    assert scale.shape == source.shape[:2]
    assert float(np.max(np.abs(luma_error))) < 1e-6
    assert facts["selected_lstar_inversion_fraction"] == 0.0
    assert facts["global_dose"] > 0.0
