from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.b0_perceptual_residual_factorization import validate_contract
from src.roll2film.density_residual_guard import (
    apply_target_density_residual_guard,
)
from src.roll2film.perceptual_residual_guard import (
    apply_target_perceptual_residual_guard,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ba0_perceptual_residual_factorization_v1.json"
)


def test_frozen_contract_binds_ao6_and_az0() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["samples"]) == 41
    assert len(validated["density_records"]) == 41
    assert config["candidate"]["lightness_strength"] == 0.15
    assert config["candidate"]["chroma_strength"] == 0.35


def test_perceptual_factorization_is_not_density_factorization() -> None:
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
    perceptual = apply_target_perceptual_residual_guard(
        source,
        target,
        lightness_strength=0.15,
        chroma_strength=0.35,
        **common,
    )
    density = apply_target_density_residual_guard(
        source,
        target,
        neutral_strength=0.15,
        opponent_strength=0.35,
        neutral_weights=np.asarray([0.2126, 0.7152, 0.0722]),
        density_floor=2.0**-16,
        **common,
    )
    assert np.max(np.abs(perceptual.output - density.output)) > 0.01
    assert np.all(
        (perceptual.output >= 0.0) & (perceptual.output <= 1.0)
    )
