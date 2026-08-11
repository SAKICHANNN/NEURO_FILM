from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.identity_slope_lstar_tone_transport import (
    select_identity_slope_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb40_identity_slope_lstar_tone_development_v1.json"


def test_cb40_contract_freezes_one_eighth_identity_slope() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB40"
    assert contract["operator"]["source_identity_fraction"] == 0.125
    assert contract["population"]["source_count_exact"] == 17
    assert len(contract["population"]["included_source_ids"]) == 17


def test_cb40_selector_preserves_order_with_flat_target_distribution() -> None:
    x = np.linspace(0.03, 0.91, 103, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    base = np.full_like(source, 0.4)
    target = base.copy()
    target[..., 0] += np.float32(0.05)
    target[..., 2] -= np.float32(0.03)
    _, _, luma_error, facts = select_identity_slope_candidate(
        source,
        base,
        target,
        boundary_epsilon=1.0 / 65535.0,
        tone_knot_count=257,
        source_identity_fraction=0.125,
        dose_grid=[1.0, 0.5, 0.0],
        maximum_gradient_ratio=10.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
    )
    assert float(np.max(np.abs(luma_error))) < 1e-6
    assert facts["selected_lstar_inversion_fraction"] == 0.0
    assert facts["source_identity_fraction"] == 0.125
