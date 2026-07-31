from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab

from src.roll2film.encoded_safe_log_chroma import (
    EncodedSafeLogChromaFilmResponse,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2bk4_quantization_regularized_log_chroma_v1.json"
)
DECISION = (
    ROOT
    / "configs/u5_r2bk4_quantization_regularized_log_chroma_decision_v1.json"
)


def _operator() -> EncodedSafeLogChromaFilmResponse:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    params = config["operator"]
    return EncodedSafeLogChromaFilmResponse(
        contrast=float(params["contrast"]),
        red_green_gain=float(params["red_green_gain"]),
        blue_yellow_gain=float(params["blue_yellow_gain"]),
        midtone_chroma_lift=float(params["midtone_chroma_lift"]),
        opponent_rotation=float(params["opponent_rotation"]),
        hue_bend=float(params["hue_bend"]),
        encoded_margin=float(params["encoded_margin"]),
        chroma_floor_encoded=float(params["chroma_floor_encoded"]),
    )


def _synthetic_population() -> np.ndarray:
    levels = np.linspace(1.0 / 255.0, 254.0 / 255.0, 13)
    cube = np.stack(
        np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1
    )
    return cube.reshape(13 * 13, 13, 3).astype(np.float64)


def _median_delta_e76(source: np.ndarray, output: np.ndarray) -> float:
    delta = rgb2lab(output) - rgb2lab(source)
    return float(np.median(np.linalg.norm(delta, axis=-1)))


def test_bk4_contract_parent_and_half_code_floor_are_frozen() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = ROOT / payload["parent"]["decision"]
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == payload[
        "parent"
    ]["decision_sha256"]
    decision = json.loads(parent.read_text(encoding="utf-8"))
    assert decision["decision"] == payload["parent"]["required_decision"]
    assert payload["operator"]["chroma_floor_encoded"] == 0.5 / 255.0
    assert payload["operator"]["strength"] == 1.0


def test_bk4_identity_neutral_endpoints_and_partition_are_exact() -> None:
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


def test_bk4_zero_to_one_code_discontinuity_is_measured() -> None:
    green = np.arange(64, 193, 8, dtype=np.float64)
    blue = np.clip(green + 55, 0, 255)
    red_zero = np.stack((np.zeros_like(green), green, blue), axis=-1) / 255.0
    red_one = np.stack((np.ones_like(green), green, blue), axis=-1) / 255.0
    output_zero = _operator().apply(red_zero[None])
    output_one = _operator().apply(red_one[None])
    delta = np.linalg.norm(
        rgb2lab(output_one) - rgb2lab(output_zero), axis=-1
    )
    gate = json.loads(CONFIG.read_text(encoding="utf-8"))["primitive_gates"]
    measured = float(np.max(delta))
    assert measured == 3.233544438520992
    assert measured > gate["maximum_zero_to_one_red_code_delta_e76"]
    assert measured < 0.04 * json.loads(CONFIG.read_text(encoding="utf-8"))[
        "mechanism_basis"
    ]["bk2_observed_zero_to_one_red_code_max_delta_e76"]


def test_bk4_style_and_encoded_boundary_gates_pass() -> None:
    source = _synthetic_population()
    output = _operator().apply(source)
    source_code = np.rint(source * 65535.0).astype(np.uint16)
    output_code = np.rint(output * 65535.0).astype(np.uint16)
    source_boundary = np.any(
        (source_code == 0) | (source_code == 65535), axis=-1
    )
    output_boundary = np.any(
        (output_code == 0) | (output_code == 65535), axis=-1
    )
    gate = json.loads(CONFIG.read_text(encoding="utf-8"))["primitive_gates"]
    assert not np.any(output_boundary & ~source_boundary)
    assert _median_delta_e76(source, output) >= gate[
        "minimum_synthetic_median_delta_e76"
    ]


def test_bk4_decision_closes_failed_continuity_gate() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    result = decision["result"]
    assert decision["decision"] == (
        "reject_half_code_regularization_continuity_gate"
    )
    assert not result["automatic_pass"]
    assert not result["continuity_gate_pass"]
    assert result["style_gate_pass"]
    assert result["encoded_boundary_gate_pass"]
    assert not result["visual_review_formally_opened"]
    assert not result["thresholds_or_strengths_changed"]
    assert not decision["training_allowed"]
    assert not decision["operator_fitting_allowed"]
