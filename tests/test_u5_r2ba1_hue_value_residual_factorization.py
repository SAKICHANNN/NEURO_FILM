from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.b0_hue_value_residual_factorization import validate_contract
from src.roll2film.hue_value_residual_guard import (
    apply_target_hue_value_residual_guard,
)
from src.roll2film.perceptual_residual_guard import (
    apply_target_perceptual_residual_guard,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ba1_hue_value_residual_factorization_v1.json"


def test_frozen_contract_binds_ao6_and_az0() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["samples"]) == 41
    assert len(validated["density_records"]) == 41
    assert config["candidate"]["neutral_saturation_floor"] == 1.0 / 64.0


def test_hue_value_is_not_perceptual_factorization() -> None:
    source = np.asarray(
        [[[0.05, 0.2, 0.8], [0.8, 0.4, 0.1]]],
        dtype=np.float64,
    )
    target = np.asarray(
        [[[0.2, 0.4, 0.6], [0.6, 0.3, 0.25]]],
        dtype=np.float64,
    )
    common = {
        "hard_boundary_epsilon_encoded_srgb": 0.5 / 255.0,
        "guard_boundary_epsilon_encoded_srgb": 1.0 / 255.0,
    }
    hue_value = apply_target_hue_value_residual_guard(
        source,
        target,
        value_strength=0.15,
        hue_saturation_strength=0.35,
        neutral_saturation_floor=1.0 / 64.0,
        **common,
    )
    perceptual = apply_target_perceptual_residual_guard(
        source,
        target,
        lightness_strength=0.15,
        chroma_strength=0.35,
        **common,
    )
    assert np.max(np.abs(hue_value.output - perceptual.output)) > 0.01
    assert np.all((hue_value.output >= 0.0) & (hue_value.output <= 1.0))
