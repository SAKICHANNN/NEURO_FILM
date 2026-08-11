from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.direct_lab_lch_transport import select_direct_lab_lch_candidate
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_photographic import _compiled_curve

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb46_direct_lab_lch_development_v1.json"
DECISION = ROOT / "configs/u5_r2cb46_direct_lab_lch_development_decision_v1.json"


def _curve():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    cb11 = json.loads((ROOT / contract["parents"]["cb11_contract_path"]).read_text())
    return _compiled_curve(load_cb6(ROOT / cb11["parents"]["cb6_contract_path"]))


def test_cb46_contract_freezes_direct_lab_and_primary_neighbor() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB46"
    assert contract["operator"]["gamut_iterations"] == 24
    assert "10.2352/EI.2025" in contract["operator"]["primary_near_neighbor"]
    assert "not independent confirmation" in contract["claim_ceiling"]


def test_cb46_neutral_ramp_is_finite_safe_and_order_preserving() -> None:
    x = np.linspace(0.02, 0.98, 257, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    candidate, scale, luma_error, facts = select_direct_lab_lch_candidate(
        source,
        source.copy(),
        curve=_curve(),
        strength=0.2,
        boundary_epsilon=1.0 / 65535.0,
        dose_grid=[1.0, 0.0],
        maximum_gradient_ratio=2.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
        gamut_iterations=24,
    )
    assert candidate.dtype == np.float32
    assert np.isfinite(candidate).all()
    assert np.min(candidate) >= 0.0 and np.max(candidate) <= 1.0
    assert facts["selected_lstar_inversion_fraction"] == 0.0
    assert facts["selected_new_boundary_fraction"] == 0.0
    assert np.max(np.abs(luma_error)) <= 0.0001
    assert scale.shape == source.shape[:-1]


def test_cb46_exact_black_white_endpoints_use_internal_neutral_rails() -> None:
    source = np.asarray([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    candidate, _, _, facts = select_direct_lab_lch_candidate(
        source,
        source.copy(),
        curve=_curve(),
        strength=0.2,
        boundary_epsilon=1.0 / 65535.0,
        dose_grid=[1.0, 0.0],
        maximum_gradient_ratio=2.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
        gamut_iterations=24,
    )
    assert np.min(candidate) > 0.0
    assert np.max(candidate) < 1.0
    assert facts["selected_new_boundary_fraction"] == 0.0


def test_cb46_out_of_gamut_chroma_is_reduced_without_rgb_clipping() -> None:
    source = np.full((1, 32, 3), 0.18, dtype=np.float32)
    target = source.copy()
    target[..., 0] = 0.99
    target[..., 1] = 0.01
    candidate, _, _, facts = select_direct_lab_lch_candidate(
        source,
        target,
        curve=_curve(),
        strength=0.2,
        boundary_epsilon=1.0 / 65535.0,
        dose_grid=[1.0, 0.0],
        maximum_gradient_ratio=100.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
        gamut_iterations=24,
    )
    assert np.min(candidate) >= 0.0 and np.max(candidate) <= 1.0
    assert facts["median_gamut_chroma_retention"] < 1.0


def test_cb46_decision_closes_lab_roundtrip_for_analytic_rgb_family() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["repeat_report_sha256_exact"] is True
    assert decision["failure"]["completed_source_count"] == 0
    assert decision["visual_review_status"] == "forbidden"
    assert decision["decision"] == (
        "close_direct_lab_roundtrip_open_analytic_luminance_eigen_affine"
    )
