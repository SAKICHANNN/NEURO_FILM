from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_photographic import _compiled_curve
from src.eval.global_affine_tone_transport import select_global_affine_tone_candidate

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb44_constrained_affine_tone_development_v1.json"


def test_cb44_contract_adds_only_nonnegative_intercept_projection() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB44"
    assert contract["operator"]["minimum_affine_intercept"] == 0.0
    assert contract["population"]["source_count_exact"] == 12
    assert "not independent confirmation" in contract["claim_ceiling"]


def test_cb44_projection_clamps_negative_fit_intercept_to_zero() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    cb11 = json.loads((ROOT / contract["parents"]["cb11_contract_path"]).read_text())
    curve = _compiled_curve(load_cb6(ROOT / cb11["parents"]["cb6_contract_path"]))
    x = np.linspace(0.1, 0.8, 257, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    target = np.asarray(0.8 * source - 0.02, dtype=np.float32)
    _, _, _, facts = select_global_affine_tone_candidate(
        source,
        target,
        curve=curve,
        strength=0.2,
        boundary_epsilon=1.0 / 65535.0,
        dose_grid=[1.0, 0.0],
        maximum_gradient_ratio=10.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
        tone_dose_grid=[1.0, 0.0],
        minimum_affine_slope=0.25,
        maximum_affine_slope=2.0,
        minimum_affine_intercept=0.0,
    )
    assert facts["fitted_affine_intercept"] == 0.0
    assert facts["selected_lstar_inversion_fraction"] == 0.0
