from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from skimage.color import rgb2lab

from src.eval.log_chroma_fresh_comparison import _safe_rich
from src.roll2film.orthogonal_perceptual_residual import (
    OrthogonalPerceptualResidual,
)
from src.roll2film.smooth_perceptual_hue_density import (
    SmoothPerceptualHueDensityResponse,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk10_orthogonal_perceptual_residual_v1.json"


def test_bk10_primitive_contract_is_frozen_before_implementation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = config["parent"]
    assert config["status"] == "primitive_contract_frozen"
    assert hashlib.sha256((ROOT / parent["decision"]).read_bytes()).hexdigest() == (
        parent["decision_sha256"]
    )
    operator = config["operator"]
    assert operator["residual_strength"] == 0.8
    assert operator["maximum_orthogonal_delta_e76"] == 12.0
    assert operator["gamut_iterations"] == 24
    assert config["fixed_inputs"]["base"]["seed"] == 20260731
    mechanism = config["mechanism_basis"]
    assert mechanism["synthetic_only_parameter_selection"]
    assert not mechanism["bk8_or_bk9_pixels_read_for_parameter_selection"]
    assert not mechanism["per_image_fit_or_routing"]
    assert not mechanism["dense_arm_weight_prediction"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["production_default_changed"]


def test_bk10_synthetic_witnesses_exceed_frozen_primitive_gates() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    witness = config["mechanism_basis"]
    gates = config["primitive_gates"]
    assert witness["prototype_17_cube_median_style_delta_e76"] >= (
        gates["minimum_17_cube_median_style_delta_e76"]
    )
    assert witness[
        "prototype_17_cube_median_increment_vs_base_delta_e76"
    ] >= gates["minimum_17_cube_median_increment_vs_base_delta_e76"]
    assert witness[
        "prototype_maximum_increment_before_gamut_delta_e76"
    ] <= gates["maximum_increment_before_gamut_delta_e76"]
    assert witness["prototype_maximum_absolute_projection_dot"] <= (
        gates["maximum_absolute_projection_dot"]
    )
    assert witness["prototype_new_uint16_boundary_fraction"] == (
        gates["new_uint16_boundary_fraction_on_17_cube"]
    )


def _cube() -> np.ndarray:
    axis = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    red, green, blue = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((red, green, blue), axis=-1).reshape(-1, 17, 3)


def _safe_rich_inputs() -> dict[str, object]:
    profile = json.loads(
        (ROOT / "configs/render_profiles/safe_rich_v1.json").read_text(
            encoding="utf-8"
        )
    )
    statistics = json.loads(
        (ROOT / "configs/film_color_stats.json").read_text(encoding="utf-8")
    )
    guardrails = json.loads(
        (ROOT / "configs/color_guardrails.json").read_text(encoding="utf-8")
    )
    style = "velvia_50"
    return {
        "safe_profile": profile["style_parameters"][style],
        "safe_statistics": statistics["styles"][style],
        "safe_guardrails": {
            **guardrails["defaults"],
            **guardrails["styles"][style],
        },
        "safe_style": style,
        "safe_seed": 20260731,
    }


def _fixed_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _cube()
    base = _safe_rich(source, _safe_rich_inputs()).astype(np.float32)
    candidate = SmoothPerceptualHueDensityResponse().apply(source)
    return source, base, candidate


def test_bk10_implementation_passes_frozen_cube_gates() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    gates = config["primitive_gates"]
    operator = OrthogonalPerceptualResidual()
    source, base, candidate = _fixed_inputs()
    output = operator.apply(source, base, candidate)
    style = np.linalg.norm(rgb2lab(output) - rgb2lab(source), axis=-1)
    increment = np.linalg.norm(rgb2lab(output) - rgb2lab(base), axis=-1)
    assert float(np.median(style)) >= (
        gates["minimum_17_cube_median_style_delta_e76"]
    )
    assert float(np.median(increment)) >= (
        gates["minimum_17_cube_median_increment_vs_base_delta_e76"]
    )
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


def test_bk10_residual_is_bounded_and_orthogonal_before_gamut() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    gates = config["primitive_gates"]
    operator = OrthogonalPerceptualResidual()
    source, base, candidate = _fixed_inputs()
    base_lab, residual = operator._orthogonal_lab_residual(
        source, base, candidate
    )
    source_lab = operator._lab(source)
    base_residual = base_lab - source_lab
    supported = np.sum(base_residual * base_residual, axis=-1) > (
        operator.projection_epsilon
    )
    dot = np.sum(residual * base_residual, axis=-1)
    assert float(np.max(np.abs(dot[supported]))) <= (
        gates["maximum_absolute_projection_dot"]
    )
    assert float(
        np.max(
            np.linalg.norm(
                operator.residual_strength * residual,
                axis=-1,
            )
        )
    ) <= gates["maximum_increment_before_gamut_delta_e76"]


def test_bk10_identity_endpoints_and_partition_are_exact() -> None:
    operator = OrthogonalPerceptualResidual()
    source, base, candidate = _fixed_inputs()
    assert np.array_equal(
        operator.apply(source, base, candidate, strength=0.0),
        base,
    )
    output = operator.apply(source, base, candidate)
    assert np.array_equal(output[0, 0, 0], source[0, 0, 0])
    assert np.array_equal(output[-1, -1, -1], source[-1, -1, -1])
    partitioned = np.concatenate(
        (
            operator.apply(source[:8], base[:8], candidate[:8]),
            operator.apply(source[8:], base[8:], candidate[8:]),
        ),
        axis=0,
    )
    assert np.array_equal(partitioned, output)


def test_bk10_rejects_invalid_inputs_and_parameters() -> None:
    with pytest.raises(ValueError):
        OrthogonalPerceptualResidual(maximum_orthogonal_delta_e76=0.0)
    operator = OrthogonalPerceptualResidual()
    source = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        operator.apply(source, source[..., :2], source)
    with pytest.raises(ValueError):
        operator.apply(source, source, source, strength=1.1)
