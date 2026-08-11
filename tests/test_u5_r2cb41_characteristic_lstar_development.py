from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.characteristic_lstar_transport import (
    select_characteristic_lstar_candidate,
)
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_minmax import _anchored_curve
from src.eval.fujifilm_characteristic_photographic import _compiled_curve

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb41_characteristic_lstar_development_v1.json"


def test_cb41_contract_reuses_exact_cb11_curve_without_histogram_fit() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB41"
    assert contract["population"]["source_count_exact"] == 17
    assert "empirical histogram maps" in contract["operator"]["forbidden"]


def test_cb41_selector_is_exactly_neutral_on_neutral_ramp() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    cb11 = json.loads((ROOT / contract["parents"]["cb11_contract_path"]).read_text())
    curve = _compiled_curve(load_cb6(ROOT / cb11["parents"]["cb6_contract_path"]))
    x = np.linspace(0.02, 0.98, 257, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    target = source.copy()
    candidate, _, luma_error, facts = select_characteristic_lstar_candidate(
        source,
        target,
        curve=curve,
        strength=float(cb11["operator"]["nominal_strength"]),
        boundary_epsilon=float(cb11["operator"]["boundary_epsilon"]),
        dose_grid=[1.0, 0.5, 0.0],
        maximum_gradient_ratio=10.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
    )
    assert candidate.dtype == np.float32
    assert float(np.max(np.abs(luma_error))) < 1e-6
    assert facts["selected_lstar_inversion_fraction"] == 0.0
    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    expected = (
        _anchored_curve(
            source_lstar.astype(np.float64) / 100.0,
            curve,
            strength=float(cb11["operator"]["nominal_strength"]),
            epsilon=float(cb11["operator"]["boundary_epsilon"]),
        )
        * 100.0
    )
    actual = linear_rgb_to_lab(candidate, working_space="linear_srgb")[..., 0]
    assert float(np.max(np.abs(actual - expected))) < 1e-3
