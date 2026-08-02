from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.real_uniform_thomas_confirmation import (
    RealUniformThomasConfirmationError,
    _model_acf,
    _model_signature,
    _validate_contract,
    evaluate_real_uniform_thomas_confirmation,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bs_real_uniform_thomas_confirmation_v1.json"


def test_model_signatures_are_finite_and_distinct() -> None:
    edges = np.geomspace(1.0 / 64.0, 0.45, 17)
    gaussian = _model_signature(edges, particle_sigma_pixels=0.8)
    thomas = _model_signature(
        edges,
        particle_sigma_pixels=0.8,
        cluster_sigma_pixels=3.0,
        mean_offspring=4.0,
    )
    assert np.all(np.isfinite(gaussian))
    assert np.all(np.isfinite(thomas))
    assert not np.array_equal(gaussian, thomas)
    assert np.linalg.norm(gaussian) == pytest.approx(1.0)
    assert np.linalg.norm(thomas) == pytest.approx(1.0)


def test_model_acf_has_unit_origin_and_finite_lags() -> None:
    lags = [[0, 1], [1, 0], [4, 0], [0, 8]]
    values = _model_acf(
        shape=(128, 128),
        lags=lags,
        particle_sigma_pixels=0.8,
        cluster_sigma_pixels=3.0,
        mean_offspring=4.0,
    )
    assert values.shape == (4,)
    assert np.all(np.isfinite(values))
    assert values[0] == pytest.approx(values[1])


def test_contract_rejects_gate_drift() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    contract["automatic_gates"][
        "minimum_confirmation_nps_median_improvement_over_gaussian"
    ] = 0.0
    with pytest.raises(RealUniformThomasConfirmationError):
        _validate_contract(contract)


@pytest.mark.skipif(
    not (ROOT / "data/external/wikimedia_uniform_grain_v1").is_dir()
    or not (ROOT / "data/external/wikimedia_bw_uniform_grain_v1").is_dir(),
    reason="exact local uniform-scan cohorts are unavailable",
)
def test_exact_real_uniform_confirmation_is_repeatable() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = evaluate_real_uniform_thomas_confirmation(contract, ROOT)
    second = evaluate_real_uniform_thomas_confirmation(contract, ROOT)
    assert first == second
    assert first["development_source_count"] == 8
    assert first["confirmation_source_count"] == 3
    assert first["checks"]["repeat_exact"] is True
