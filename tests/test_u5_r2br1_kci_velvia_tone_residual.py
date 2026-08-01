from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2br1_kci_velvia_tone_residual import (
    CONFIG_SHA256,
    load_config,
)
from src.real_film.kci_velvia_simulation_residual import evaluate_tone_residual


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2br1_kci_velvia_tone_residual_v1.json"
PARENT = ROOT / "configs/u5_r2br0_kci_velvia_simulation_residual_v1.json"


def test_contract_preserves_parent_and_claim_boundary() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert config["split"]["fold_count"] == 6
    assert config["parent"]["config_sha256"] == (
        "bd7edf5c86b3f1d50d639a8891cef6e8c52437216e8932f183da659dfe9556c3"
    )
    assert not config["production_integration_allowed"]
    assert not config["stock_response_claim_allowed"]
    assert not config["calibrated_reference_claim_allowed"]


def test_frozen_neutral_observation_has_six_disjoint_folds() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    digital = np.asarray(parent["extraction"]["digital_patch_rgb_u8"], dtype=np.uint8)
    film = np.asarray(parent["extraction"]["film_patch_rgb_u8"], dtype=np.uint8)
    result = evaluate_tone_residual(digital, film, config)
    assert len(result["folds"]) == 6
    seen: list[int] = []
    for fold in result["folds"]:
        assert len(fold["development_indices"]) == 5
        assert len(fold["confirmation_indices"]) == 1
        assert not set(fold["development_indices"]) & set(
            fold["confirmation_indices"]
        )
        seen.extend(fold["confirmation_indices"])
    assert sorted(seen) == list(range(6))
    assert result["aggregate"]["identity"][
        "mean_confirmation_absolute_lstar_error"
    ] == pytest.approx(8.559095003403819)
    assert result["aggregate"]["monotone_pchip"][
        "mean_confirmation_absolute_lstar_error"
    ] == pytest.approx(2.2153300634730018)
    assert result["aggregate"]["monotone_pchip"][
        "maximum_confirmation_absolute_lstar_error"
    ] == pytest.approx(5.523859455302297)
    assert result["aggregate"]["monotone_pchip"][
        "gain_over_gamma"
    ] == pytest.approx(0.36963075637368203)
    assert result["aggregate"]["monotone_pchip"][
        "gain_over_affine"
    ] == pytest.approx(0.7202145606491268)
    assert result["automatic_pass"]
    assert result["selected_model"] == "monotone_pchip"
    assert result["decision"] == (
        "retain_controlled_monotone_tone_residual_for_photo_stress"
    )


def test_evaluator_rejects_invalid_shape() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="RGB8 24x3"):
        evaluate_tone_residual(np.zeros((6, 3)), np.zeros((6, 3)), config)
