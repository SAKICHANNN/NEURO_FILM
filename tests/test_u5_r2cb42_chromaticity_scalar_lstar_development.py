from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.chromaticity_scalar_lstar_transport import (
    select_chromaticity_scalar_candidate,
)
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_photographic import _compiled_curve

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb42_chromaticity_scalar_lstar_development_v1.json"


def test_cb42_contract_changes_execution_not_curve_or_chroma_target() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB42"
    assert contract["population"]["source_count_exact"] == 9
    assert "positive per-pixel scalar" in contract["operator"]["tone_execution"]
    assert contract["automatic_gates"]["maximum_tone_lstar_error"] == 0.001


def test_cb42_scalar_solve_preserves_rgb_ratios_on_neutral_target() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    cb11 = json.loads((ROOT / contract["parents"]["cb11_contract_path"]).read_text())
    curve = _compiled_curve(load_cb6(ROOT / cb11["parents"]["cb6_contract_path"]))
    x = np.linspace(0.05, 0.65, 129, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    target = source.copy()
    candidate, _, luma_error, facts = select_chromaticity_scalar_candidate(
        source,
        target,
        curve=curve,
        strength=float(cb11["operator"]["nominal_strength"]),
        boundary_epsilon=float(cb11["operator"]["boundary_epsilon"]),
        dose_grid=[1.0, 0.0],
        maximum_gradient_ratio=10.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
    )
    assert candidate.dtype == np.float32
    assert float(np.max(np.abs(luma_error))) < 1e-6
    assert facts["selected_lstar_inversion_fraction"] == 0.0
    assert facts["scalar_gamut_limited_fraction"] == 0.0
    assert np.array_equal(candidate[..., 0], candidate[..., 1])
    assert np.array_equal(candidate[..., 1], candidate[..., 2])
