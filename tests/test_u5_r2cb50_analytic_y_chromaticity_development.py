from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.analytic_y_chromaticity_transport import (
    _bounded_same_y_reconstruction,
    _normalized_zero_y_chromaticity,
    select_analytic_y_chromaticity_candidate,
)
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_photographic import _compiled_curve

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb50_analytic_y_chromaticity_development_v1.json"


def _curve():
    cb11 = json.loads(
        (
            ROOT / "configs/u5_r2cb11_fujifilm_characteristic_luma_chroma_v1.json"
        ).read_text(encoding="utf-8")
    )
    return _compiled_curve(load_cb6(ROOT / cb11["parents"]["cb6_contract_path"]))


def test_cb50_contract_freezes_distinct_factorization() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "U5.R2CB50"
    assert "divided by source or target Y" in payload["operator"]["chromaticity"]
    assert "no channel clipping" in payload["operator"]["gamut"]


def test_cb50_normalized_chromaticity_has_zero_y() -> None:
    rng = np.random.default_rng(50)
    source = rng.uniform(0.01, 0.99, (11, 17, 3)).astype(np.float32)
    chroma = _normalized_zero_y_chromaticity(source, 1 / 65535)
    assert np.max(np.abs(chroma @ LEGACY_LAB_Y_WEIGHTS)) <= 1e-15


def test_cb50_reconstruction_preserves_y_and_cube() -> None:
    rng = np.random.default_rng(51)
    y = rng.uniform(0.02, 0.98, (13, 19))
    chroma = rng.normal(0, 3, (13, 19, 3))
    chroma -= (chroma @ LEGACY_LAB_Y_WEIGHTS)[..., None]
    candidate, scale, error = _bounded_same_y_reconstruction(
        y, chroma, boundary_epsilon=1 / 65535
    )
    assert np.min(candidate) > 0 and np.max(candidate) < 1
    assert np.min(scale) >= 0 and np.max(scale) <= 1
    assert np.max(np.abs(error)) <= 1e-6


def test_cb50_candidate_is_safe_and_nontrivial() -> None:
    axis = np.linspace(0.02, 0.9, 512, dtype=np.float32).reshape(16, 32)
    source = np.stack((axis, axis * 0.8, axis * 0.6), axis=-1).astype(np.float32)
    target = np.stack((axis * 0.6, axis * 0.9, axis * 0.4), axis=-1).astype(np.float32)
    candidate, scale, error, facts = select_analytic_y_chromaticity_candidate(
        source,
        target,
        curve=_curve(),
        strength=0.2,
        boundary_epsilon=1 / 510,
        dose_grid=[1.0, 0.5, 0.0],
        maximum_gradient_ratio=2.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
    )
    assert np.min(candidate) >= 0 and np.max(candidate) <= 1
    assert np.max(np.abs(error)) <= 1e-6
    assert facts["selected_lstar_inversion_fraction"] == 0
    assert facts["global_dose"] > 0
    assert np.median(scale) > 0
