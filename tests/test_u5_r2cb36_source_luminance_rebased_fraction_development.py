from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.source_luminance_rebased_fraction_transport import (
    select_source_luminance_rebased_candidate,
    source_luminance_rebased_target,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u5_r2cb36_source_luminance_rebased_fraction_development_v1.json"
)


def test_cb36_contract_freezes_new_population_and_source_luminance_base() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB36"
    assert contract["population"]["source_count_exact"] == 11
    assert contract["population"]["exact_decoded_sha_overlap_with_cb27_cb28_cb34_cb35"] == 0
    assert contract["operator"]["execution_base"] == "exact source linear RGB"
    assert contract["automatic_gates"]["maximum_adjacent_lstar_gradient_sign_inversion_fraction"] == 0.0


def test_cb36_rebase_places_target_chroma_on_source_luminance() -> None:
    rng = np.random.default_rng(20260812)
    source = rng.uniform(0.1, 0.9, size=(23, 17, 3)).astype(np.float32)
    target = rng.uniform(-0.2, 1.2, size=source.shape).astype(np.float32)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    rebased = source_luminance_rebased_target(source, target, weights=weights)
    source_y = np.sum(source.astype(np.float64) * weights, axis=-1)
    rebased_y = np.sum(rebased.astype(np.float64) * weights, axis=-1)
    assert float(np.max(np.abs(rebased_y - source_y))) < 1e-7


def test_cb36_selector_preserves_source_order_on_chromatic_target() -> None:
    x = np.linspace(0.05, 0.85, 31, dtype=np.float32)
    source = np.repeat(x[None, :, None], 19, axis=0)
    source = np.repeat(source, 3, axis=2)
    target = source.copy()
    target[..., 0] += np.float32(0.12)
    target[..., 1] -= np.float32(0.03)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    candidate, scale, luma_error, facts = select_source_luminance_rebased_candidate(
        source,
        target,
        weights=weights,
        boundary_epsilon=1.0 / 65535.0,
        dose_grid=[1.0, 0.5, 0.0],
        maximum_gradient_ratio=1.35,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
    )
    assert candidate.shape == source.shape
    assert scale.shape == source.shape[:2]
    assert float(np.max(np.abs(luma_error))) < 1e-7
    assert facts["selected_lstar_inversion_fraction"] == 0.0
    assert facts["global_dose"] > 0.0
