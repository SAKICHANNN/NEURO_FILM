from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from skimage.color import rgb2lab

from src.roll2film.encoded_safe_log_chroma import (
    EncodedSafeLogChromaFilmResponse,
)
from src.roll2film.log_chroma_film_response import LogChromaFilmResponse
from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_population() -> np.ndarray:
    levels = np.linspace(1.0 / 255.0, 254.0 / 255.0, 13)
    cube = np.stack(
        np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1
    )
    return cube.reshape(13 * 13, 13, 3).astype(np.float64)


def _median_delta_e76(source: np.ndarray, output: np.ndarray) -> float:
    delta = rgb2lab(output) - rgb2lab(source)
    return float(np.median(np.linalg.norm(delta, axis=-1)))


def test_frozen_contract_matches_v2_defaults() -> None:
    payload = json.loads(
        (ROOT / "configs/u5_r2bk2_encoded_safe_log_chroma_v1.json").read_text(
            encoding="utf-8"
        )
    )
    params = payload["operator"]
    op = EncodedSafeLogChromaFilmResponse()
    for key in (
        "contrast",
        "red_green_gain",
        "blue_yellow_gain",
        "midtone_chroma_lift",
        "opponent_rotation",
        "hue_bend",
        "encoded_margin",
    ):
        assert params[key] == getattr(op, key)
    parent = json.loads(
        (ROOT / payload["parent"]["decision"]).read_text(encoding="utf-8")
    )
    assert parent["decision"] == payload["parent"]["required_decision"]


def test_v2_identity_neutral_axis_endpoints_and_partition() -> None:
    op = EncodedSafeLogChromaFilmResponse()
    neutral = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    rgb = np.repeat(neutral[:, None], 3, axis=1)
    output = op.apply(rgb)
    assert np.array_equal(op.apply(rgb, strength=0.0), rgb)
    assert np.max(np.ptp(output, axis=1)) <= 2e-15
    assert np.array_equal(output[[0, -1]], rgb[[0, -1]])
    source = _synthetic_population().astype(np.float32)
    full = op.apply(source)
    rows = np.concatenate(
        (op.apply(source[:71]), op.apply(source[71:129]), op.apply(source[129:]))
    )
    assert np.array_equal(full, rows)


def test_v2_encoded_boundary_and_synthetic_style_gates() -> None:
    source = _synthetic_population()
    v2 = EncodedSafeLogChromaFilmResponse().apply(source)
    v1_linear = LogChromaFilmResponse().apply(encoded_srgb_to_linear(source))
    v1 = linear_srgb_to_encoded(v1_linear)
    source_code = np.rint(source * 65535.0).astype(np.uint16)
    output_code = np.rint(v2 * 65535.0).astype(np.uint16)
    source_boundary = np.any(
        (source_code == 0) | (source_code == 65535), axis=-1
    )
    output_boundary = np.any(
        (output_code == 0) | (output_code == 65535), axis=-1
    )
    assert not np.any(output_boundary & ~source_boundary)
    v2_style = _median_delta_e76(source, v2)
    v1_style = _median_delta_e76(source, v1)
    assert v2_style >= 5.0
    assert v2_style / v1_style >= 1.1


def test_v2_invalid_inputs_fail_closed() -> None:
    op = EncodedSafeLogChromaFilmResponse()
    with pytest.raises(ValueError):
        op.apply(np.asarray([[[np.nan, 0.0, 0.0]]], dtype=np.float32))
    with pytest.raises(ValueError):
        op.apply(np.zeros((2, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        op.apply(np.zeros((2, 2, 3), dtype=np.float32), strength=-0.01)


def test_v2_exact_zero_individual_channels_stay_finite() -> None:
    source = np.asarray(
        [
            [[0.0, 0.4, 0.8], [0.7, 0.0, 0.2]],
            [[0.3, 0.9, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    output = EncodedSafeLogChromaFilmResponse().apply(source)
    assert np.all(np.isfinite(output))
    assert np.all(output >= 0.0)
    assert np.all(output <= 1.0)
