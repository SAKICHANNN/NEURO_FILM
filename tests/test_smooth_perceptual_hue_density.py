from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from skimage.color import rgb2lab

from src.roll2film.smooth_perceptual_hue_density import (
    SmoothPerceptualHueDensityResponse,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk7_smooth_perceptual_hue_density_v1.json"


def test_bk7_primitive_contract_is_frozen_before_implementation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = config["parent"]
    assert config["status"] == "primitive_contract_frozen"
    assert hashlib.sha256((ROOT / parent["decision"]).read_bytes()).hexdigest() == (
        parent["decision_sha256"]
    )
    assert config["operator"]["strength"] == 1.0
    assert config["operator"]["gamut_iterations"] == 24
    assert config["mechanism_basis"]["synthetic_only_parameter_selection"]
    assert not config["mechanism_basis"]["bk6_pixels_read_for_parameter_selection"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["production_default_changed"]
    assert not config["stock_or_authenticity_claim_allowed"]


def test_bk7_synthetic_witnesses_exceed_frozen_primitive_gates() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    witness = config["mechanism_basis"]
    gates = config["primitive_gates"]
    assert witness["prototype_17_cube_median_style_delta_e76"] >= (
        gates["minimum_17_cube_median_style_delta_e76"]
    )
    assert witness["prototype_zero_to_one_axis_max_delta_e76"] <= (
        gates["maximum_zero_to_one_axis_delta_e76"]
    )
    assert witness["prototype_maximum_neutral_axis_channel_range"] <= (
        gates["maximum_neutral_axis_channel_range"]
    )
    assert witness["prototype_new_uint16_boundary_fraction"] == (
        gates["new_uint16_boundary_fraction_on_17_cube"]
    )


def _cube() -> np.ndarray:
    axis = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    red, green, blue = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((red, green, blue), axis=-1).reshape(-1, 17, 3)


def test_bk7_implementation_passes_frozen_cube_gate() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    witness = config["mechanism_basis"]
    gates = config["primitive_gates"]
    source = _cube()
    output = SmoothPerceptualHueDensityResponse().apply(source)
    delta = np.linalg.norm(rgb2lab(output) - rgb2lab(source), axis=-1)
    median_delta = float(np.median(delta))
    assert median_delta >= gates["minimum_17_cube_median_style_delta_e76"]
    assert median_delta <= witness["prototype_17_cube_median_style_delta_e76"]
    source_code = np.rint(source * 65535.0).astype(np.uint16)
    output_code = np.rint(output * 65535.0).astype(np.uint16)
    source_boundary = np.any(
        (source_code == 0) | (source_code == 65535), axis=-1
    )
    output_boundary = np.any(
        (output_code == 0) | (output_code == 65535), axis=-1
    )
    assert np.mean(output_boundary & ~source_boundary) == (
        gates["new_uint16_boundary_fraction_on_17_cube"]
    )


def test_bk7_identity_endpoints_and_partition_are_exact() -> None:
    operator = SmoothPerceptualHueDensityResponse()
    source = _cube()
    assert np.array_equal(operator.apply(source, strength=0.0), source)
    output = operator.apply(source)
    assert np.array_equal(output[0, 0, 0], source[0, 0, 0])
    assert np.array_equal(output[-1, -1, -1], source[-1, -1, -1])
    partitioned = np.concatenate(
        (operator.apply(source[:8]), operator.apply(source[8:])),
        axis=0,
    )
    assert np.array_equal(partitioned, output)


def test_bk7_zero_to_one_and_neutral_axis_gates_pass() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    gates = config["primitive_gates"]
    operator = SmoothPerceptualHueDensityResponse()
    axis = np.arange(0.0, 256.0, 16.0, dtype=np.float32)
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    for channel in range(3):
        for first in axis:
            for second in axis:
                zero = np.asarray((first, second, first), dtype=np.float32)
                zero[channel] = 0.0
                one = zero.copy()
                one[channel] = 1.0
                lower.append(zero)
                upper.append(one)
    low_output = operator.apply(
        (np.stack(lower) / 255.0)[:, None, :]
    )
    high_output = operator.apply(
        (np.stack(upper) / 255.0)[:, None, :]
    )
    delta = np.linalg.norm(
        rgb2lab(high_output) - rgb2lab(low_output),
        axis=-1,
    )
    assert float(np.max(delta)) <= gates[
        "maximum_zero_to_one_axis_delta_e76"
    ]

    gray = np.linspace(0.0, 1.0, 257, dtype=np.float32)
    neutral = np.stack((gray, gray, gray), axis=-1)[:, None, :]
    neutral_output = operator.apply(neutral)
    assert float(np.max(np.ptp(neutral_output, axis=-1))) <= gates[
        "maximum_neutral_axis_channel_range"
    ]


def test_bk7_rejects_invalid_inputs_and_parameters() -> None:
    with pytest.raises(ValueError):
        SmoothPerceptualHueDensityResponse(tone_power=0.0)
    operator = SmoothPerceptualHueDensityResponse()
    with pytest.raises(ValueError):
        operator.apply(np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        operator.apply(np.zeros((3, 3, 3), dtype=np.float32), strength=1.1)
