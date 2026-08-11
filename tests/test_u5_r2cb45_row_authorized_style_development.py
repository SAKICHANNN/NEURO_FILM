from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.row_authorized_style_transport import (
    select_row_authorized_style_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb45_row_authorized_style_development_v1.json"
DECISION = ROOT / "configs/u5_r2cb45_row_authorized_style_development_decision_v1.json"


def _selector(source: np.ndarray, base: np.ndarray, target: np.ndarray, **overrides: float):
    kwargs = {
        "safe_base_linear": base,
        "curve": object(),
        "strength": 0.2,
        "boundary_epsilon": 1.0 / 65535.0,
        "dose_grid": [1.0, 0.0],
        "maximum_gradient_ratio": 10.0,
        "maximum_lstar_inversion_fraction": 0.0,
        "lstar_order_epsilon": 0.0001,
    }
    kwargs.update(overrides)
    return select_row_authorized_style_candidate(source, target, **kwargs)


def test_cb45_contract_freezes_atomic_row_fallback() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB45"
    assert contract["automatic_gates"]["minimum_authorized_source_count"] == 10
    assert "per-pixel repair" in contract["operator"]["forbidden"]


def test_cb45_authorizes_a_safe_whole_row() -> None:
    x = np.linspace(0.1, 0.7, 64, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    base = source.copy()
    target = source.copy()
    candidate, scale, luma_error, facts = _selector(source, base, target)
    assert facts["row_authorized"] is True
    assert facts["row_veto_reasons"] == []
    assert np.array_equal(candidate, source)
    assert np.max(np.abs(luma_error)) == 0.0
    assert scale.shape == source.shape[:-1]


def test_cb45_rejects_entire_row_on_order_failure() -> None:
    x = np.linspace(0.1, 0.7, 64, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    base = source.copy()
    target = source.copy()
    candidate, scale, luma_error, facts = _selector(
        source,
        base,
        target,
        maximum_lstar_inversion_fraction=-1.0,
    )
    assert facts["row_authorized"] is False
    assert "lstar_order" in facts["row_veto_reasons"]
    assert np.array_equal(candidate, source)
    assert np.count_nonzero(scale) == 0
    assert np.count_nonzero(luma_error) == 0


def test_cb45_decision_closes_row_fallback_for_direct_lab_construction() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["repeat_report_sha256_exact"] is True
    assert decision["metrics"]["authorized_source_count"] == 2
    assert decision["metrics"]["lstar_order_veto_count"] == 10
    assert decision["visual_review_status"] == "forbidden"
    assert decision["decision"] == (
        "close_row_authorized_style_open_direct_perceptual_lightness_chroma_construction"
    )
