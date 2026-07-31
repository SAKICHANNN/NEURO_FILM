from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from skimage.color import rgb2lab

from src.roll2film.bounded_opponent_response import (
    BoundedOpponentFilmResponse,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk5_bounded_opponent_response_v1.json"


def _operator() -> BoundedOpponentFilmResponse:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    params = config["operator"]
    return BoundedOpponentFilmResponse(
        contrast=float(params["contrast"]),
        red_green_gain=float(params["red_green_gain"]),
        blue_yellow_gain=float(params["blue_yellow_gain"]),
        midtone_chroma_lift=float(params["midtone_chroma_lift"]),
        opponent_rotation=float(params["opponent_rotation"]),
        hue_bend=float(params["hue_bend"]),
        encoded_margin=float(params["encoded_margin"]),
    )


def _synthetic_population() -> np.ndarray:
    levels = np.linspace(1.0 / 255.0, 254.0 / 255.0, 13)
    cube = np.stack(
        np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1
    )
    return cube.reshape(13 * 13, 13, 3).astype(np.float64)


def test_bk5_contract_parent_and_defaults_are_frozen() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = ROOT / payload["parent"]["decision"]
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == payload[
        "parent"
    ]["decision_sha256"]
    decision = json.loads(parent.read_text(encoding="utf-8"))
    assert decision["decision"] == payload["parent"]["required_decision"]
    op = _operator()
    for key in (
        "contrast",
        "red_green_gain",
        "blue_yellow_gain",
        "midtone_chroma_lift",
        "opponent_rotation",
        "hue_bend",
        "encoded_margin",
    ):
        assert payload["operator"][key] == getattr(op, key)


def test_bk5_identity_neutral_endpoints_and_partition_are_exact() -> None:
    op = _operator()
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


def test_bk5_zero_to_one_code_and_style_gates_pass() -> None:
    op = _operator()
    gate = json.loads(CONFIG.read_text(encoding="utf-8"))["primitive_gates"]
    green = np.arange(64, 193, 8, dtype=np.float64)
    blue = np.clip(green + 55, 0, 255)
    red_zero = np.stack((np.zeros_like(green), green, blue), axis=-1) / 255.0
    red_one = np.stack((np.ones_like(green), green, blue), axis=-1) / 255.0
    delta = np.linalg.norm(
        rgb2lab(op.apply(red_one[None]))
        - rgb2lab(op.apply(red_zero[None])),
        axis=-1,
    )
    assert float(np.max(delta)) <= gate[
        "maximum_zero_to_one_red_code_delta_e76"
    ]
    source = _synthetic_population()
    output = op.apply(source)
    style = float(
        np.median(
            np.linalg.norm(rgb2lab(output) - rgb2lab(source), axis=-1)
        )
    )
    assert style >= gate["minimum_synthetic_median_delta_e76"]
    source_code = np.rint(source * 65535.0).astype(np.uint16)
    output_code = np.rint(output * 65535.0).astype(np.uint16)
    source_boundary = np.any(
        (source_code == 0) | (source_code == 65535), axis=-1
    )
    output_boundary = np.any(
        (output_code == 0) | (output_code == 65535), axis=-1
    )
    assert not np.any(output_boundary & ~source_boundary)


def test_bk5_invalid_inputs_fail_closed() -> None:
    op = _operator()
    with pytest.raises(ValueError):
        op.apply(np.asarray([[[np.nan, 0.0, 0.0]]], dtype=np.float32))
    with pytest.raises(ValueError):
        op.apply(np.zeros((2, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        op.apply(np.zeros((2, 2, 3), dtype=np.float32), strength=1.01)
