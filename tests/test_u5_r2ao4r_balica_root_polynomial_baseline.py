from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ao4r_balica_root_polynomial_baseline import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.balica_root_polynomial_baseline import (
    _fit_polynomial,
    evaluate_balica_root_polynomial,
    gamma_prophoto_to_linear_srgb,
    linear_srgb_to_gamma_prophoto,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao4r_balica_root_polynomial_baseline_v1.json"


def _config() -> dict:
    return load_config(CONFIG, expected_sha256=CONFIG_SHA256)


def test_prophoto_roundtrip_extended_values() -> None:
    config = _config()
    rng = np.random.default_rng(20260801)
    values = rng.uniform(-0.1, 1.1, size=(1000, 3))
    replay = gamma_prophoto_to_linear_srgb(
        linear_srgb_to_gamma_prophoto(values, config), config
    )
    assert np.max(np.abs(replay - values)) < 2e-14


def test_root_polynomial_recovers_synthetic_gamma_domain_operator() -> None:
    config = json.loads(json.dumps(_config()))
    config["models"]["ridge_lambda"] = 1e-12
    rng = np.random.default_rng(91)
    source = rng.uniform(0.05, 0.95, size=(200, 3))
    encoded = linear_srgb_to_gamma_prophoto(source, config)
    target_encoded = encoded @ np.asarray(
        [[0.93, 0.04, 0.03], [0.02, 0.95, 0.03], [0.03, 0.04, 0.93]]
    ).T
    target = gamma_prophoto_to_linear_srgb(target_encoded, config)
    operator = _fit_polynomial(source, target, config, root=True)
    assert np.sqrt(np.mean(np.square(operator.apply(source) - target))) < 3e-8


def test_config_hash_and_claim_boundary_fail_closed(tmp_path: Path) -> None:
    config = _config()
    assert config["external_method"]["dataset_publicly_downloadable"] is False
    assert config["image_rendering_allowed"] is False
    config["models"]["hard_output_clipping"] = True
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(path, expected_sha256=CONFIG_SHA256)


def test_incomplete_dataset_rejected() -> None:
    config = _config()
    with pytest.raises(ValueError, match="exact chart and palette"):
        evaluate_balica_root_polynomial(
            {"velvia_chart": (np.zeros((24, 3)), np.zeros((24, 3)))}, config
        )
