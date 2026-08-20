from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2repid13_paired_style_latent import _summarize
from src.eval.repid_paired_style_latent import (
    encode_pca,
    fit_pca,
    predict_factorized,
    spatial_descriptor,
    train_factorized_models,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2repid13_paired_style_latent_v1.json"


def _descriptor_config() -> dict:
    return {
        "maximum_side": 32,
        "spatial_grid_rows": 2,
        "spatial_grid_columns": 2,
        "include_global_cell": True,
        "rgb_quantiles": [0.25, 0.5, 0.75],
        "luma_quantiles": [0.25, 0.5, 0.75],
        "chromaticity_quantiles": [0.25, 0.5, 0.75],
        "chroma_quantiles": [0.25, 0.5, 0.75],
        "epsilon": 1e-8,
    }


def test_contract_freezes_new_information_and_no_product_output() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["latent"]["teacher"] == "winner descriptor minus same-scene original descriptor"
    assert config["latent"]["student_input"] == "winner after-reference descriptor only"
    assert config["latent"]["direct_final_rgb_prediction"] is False
    assert config["roles"]["calibration_original_reads_before_prediction_freeze"] == 0
    assert config["roles"]["sealed_reads"] == 0
    assert config["primary_method_basis"]["url"] == "https://arxiv.org/abs/2602.17044"


def test_spatial_descriptor_is_repeat_exact_and_position_sensitive() -> None:
    yy, xx = np.mgrid[0:40, 0:60]
    image = np.stack((xx / 59.0, yy / 39.0, (xx + yy) / 98.0), axis=-1).astype(np.float32)
    first = spatial_descriptor(image, _descriptor_config())
    second = spatial_descriptor(image.copy(), _descriptor_config())
    flipped = spatial_descriptor(image[:, ::-1].copy(), _descriptor_config())
    assert np.array_equal(first, second)
    assert first.shape == (5 * 21,)
    assert not np.array_equal(first, flipped)


def test_spatial_descriptor_rejects_invalid_pixels() -> None:
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        spatial_descriptor(image, _descriptor_config())


def test_pca_sign_and_factorized_prediction_are_repeat_exact() -> None:
    rng = np.random.default_rng(17)
    after = rng.normal(size=(24, 10))
    delta = rng.normal(size=(24, 10))
    params = delta[:, :4] @ rng.normal(size=(4, 12))
    first = fit_pca(delta, 4)
    second = fit_pca(delta.copy(), 4)
    assert np.array_equal(first["components"], second["components"])
    for component in first["components"]:
        assert component[np.argmax(np.abs(component))] >= 0.0
    model = train_factorized_models(
        after,
        delta,
        params,
        component_count=4,
        student_alpha=10.0,
        operator_alpha=10.0,
        cyclic_shift=1,
    )
    predicted = predict_factorized(model, after[:3])
    replay = predict_factorized(model, after[:3].copy())
    assert set(predicted) == {
        "candidate_latent",
        "candidate_parameters",
        "direct_parameters",
        "permuted_parameters",
    }
    assert all(np.array_equal(predicted[key], replay[key]) for key in predicted)
    assert encode_pca(first, delta).shape == (24, 4)


def test_summary_requires_candidate_to_beat_direct_control() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = []
    for index in range(24):
        rows.append(
            {
                "candidate_improvement": 0.3,
                "gain_vs_direct": -0.01 if index < 12 else 0.01,
                "gain_vs_global": 0.2,
                "gain_vs_permuted": 0.2,
                "paired_latent_oracle_improvement": 0.3,
                "candidate_output_delta_e_oklab": 0.02,
                "candidate_new_exact_boundary_fraction": 0.0,
                "candidate_p999_gradient_ratio": 1.0,
                "candidate_matrix": {
                    "determinant": 1.0,
                    "condition_number": 1.0,
                    "minimum_singular_value": 1.0,
                },
                "latent_squared_error": 0.01,
            }
        )
    _, gates = _summarize(rows, config)
    assert gates["beat_direct_rate"] is False
    assert gates["gain_vs_direct_median"] is False
