from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.rec2020_source_anchored_interior import (
    CONTRACT_SHA256,
    Rec2020InteriorError,
    load_contract,
    source_anchored_interior_residual,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c3_rec2020_source_anchored_interior_v1.json"


def test_contract_identity_and_structure_are_frozen() -> None:
    contract = load_contract(CONTRACT)
    assert CONTRACT_SHA256 == "9b4996b8aa383081775cb000c079ef35390a5f38022a9f7f685199aa5218527b"
    assert contract["experiment_id"] == "U1.4C3"
    assert contract["mechanism"]["hard_clipping_allowed"] is False
    assert contract["mechanism"]["same_cohort_rescue_allowed"] is False


def test_analytical_scale_preserves_direction_and_interval() -> None:
    source = np.asarray([[[0.2, 0.5, 0.8], [0.0, 1.0, 0.4]]], dtype=np.float32)
    candidate = np.asarray([[[1.0, 0.0, 0.95], [0.8, 0.2, 1.0]]], dtype=np.float32)
    margin = 2.0 / 65535.0
    output, scale = source_anchored_interior_residual(source, candidate, margin=margin)
    residual = candidate.astype(np.float64) - source.astype(np.float64)
    reconstructed = source.astype(np.float64) + scale[..., None].astype(np.float64) * residual
    np.testing.assert_allclose(output, reconstructed.astype(np.float32), atol=1e-7, rtol=0.0)
    lower = np.minimum(source, margin)
    upper = np.maximum(source, 1.0 - margin)
    assert np.all(output >= lower - 2e-7)
    assert np.all(output <= upper + 2e-7)
    assert np.all((scale >= 0.0) & (scale <= 1.0))


def test_source_boundary_is_not_forced_inward() -> None:
    source = np.asarray([[[0.0, 1.0, 0.5]]], dtype=np.float32)
    candidate = np.asarray([[[0.0, 1.0, 0.6]]], dtype=np.float32)
    output, scale = source_anchored_interior_residual(source, candidate, margin=2.0 / 65535.0)
    np.testing.assert_array_equal(output, candidate)
    np.testing.assert_array_equal(scale, np.ones((1, 1), dtype=np.float32))


@pytest.mark.parametrize("margin", [0.0, -0.1, 0.5, float("nan")])
def test_invalid_margin_fails_closed(margin: float) -> None:
    source = np.full((2, 2, 3), 0.5, dtype=np.float32)
    with pytest.raises(Rec2020InteriorError):
        source_anchored_interior_residual(source, source, margin=margin)


def test_nonfinite_or_out_of_gamut_input_fails_closed() -> None:
    source = np.full((2, 2, 3), 0.5, dtype=np.float32)
    invalid = source.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(Rec2020InteriorError):
        source_anchored_interior_residual(source, invalid, margin=2.0 / 65535.0)
    invalid = source.copy()
    invalid[0, 0, 0] = 1.01
    with pytest.raises(Rec2020InteriorError):
        source_anchored_interior_residual(source, invalid, margin=2.0 / 65535.0)
