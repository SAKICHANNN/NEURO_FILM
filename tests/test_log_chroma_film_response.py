from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.roll2film.log_chroma_film_response import LogChromaFilmResponse


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_contract_matches_implementation_defaults() -> None:
    payload = json.loads(
        (ROOT / "configs/u5_r2bk0_log_chroma_film_response_v1.json").read_text(
            encoding="utf-8"
        )
    )
    params = payload["operator"]
    op = LogChromaFilmResponse()
    assert params["contrast"] == op.contrast
    assert params["chroma_gain"] == op.chroma_gain
    assert params["midtone_chroma_lift"] == op.midtone_chroma_lift
    assert params["opponent_rotation"] == op.opponent_rotation
    assert params["margin"] == op.margin
    assert params["strength"] == 1.0
    assert (ROOT / payload["parent"]["decision"]).is_file()


def test_identity_neutral_axis_and_endpoints() -> None:
    op = LogChromaFilmResponse()
    neutral = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    rgb = np.repeat(neutral[:, None], 3, axis=1)
    out = op.apply(rgb)
    assert np.array_equal(op.apply(rgb, strength=0.0), rgb)
    assert np.max(np.ptp(out, axis=1)) <= 2e-15
    assert np.array_equal(out[[0, -1]], rgb[[0, -1]])


def test_cube_margin_and_partition_exact() -> None:
    rng = np.random.default_rng(20260731)
    rgb = rng.random((37, 29, 3), dtype=np.float32)
    op = LogChromaFilmResponse()
    full = op.apply(rgb)
    rows = np.concatenate([op.apply(rgb[:11]), op.apply(rgb[11:23]), op.apply(rgb[23:])])
    assert np.array_equal(full, rows)
    assert np.all(full >= 0.0)
    assert np.all(full <= 1.0)
    assert np.median(np.abs(full.astype(np.float64) - rgb)) > 0.005


def test_invalid_inputs_fail_closed() -> None:
    op = LogChromaFilmResponse()
    with pytest.raises(ValueError):
        op.apply(np.asarray([[[np.nan, 0.0, 0.0]]], dtype=np.float32))
    with pytest.raises(ValueError):
        op.apply(np.zeros((2, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        op.apply(np.zeros((2, 2, 3), dtype=np.float32), strength=1.01)
