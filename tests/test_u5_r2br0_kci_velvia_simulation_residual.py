from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2br0_kci_velvia_simulation_residual import (
    CONFIG_SHA256,
    load_config,
)
from src.real_film.kci_velvia_simulation_residual import (
    evaluate_residual,
    extract_exact_pair,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2br0_kci_velvia_simulation_residual_v1.json"
DATA = ROOT / "data/real_film/kci_velvia_simulation_2025_v1/derived"


def test_contract_is_controlled_development_only() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert config["split"]["fold_count"] == 6
    assert config["extraction"]["chromatic_patch_indices"] == list(range(18))
    assert not config["production_integration_allowed"]
    assert not config["stock_response_claim_allowed"]
    assert not config["calibrated_reference_claim_allowed"]


def test_exact_plot_extraction_matches_frozen_patches() -> None:
    digital = DATA / "digital_velvia_imatest_plot.png"
    film = DATA / "film_velvia_imatest_plot.png"
    if not digital.exists() or not film.exists():
        pytest.skip("exact ignored KCI plot images are not installed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = extract_exact_pair(digital, film, config)
    assert result["digital_rgb_u8"].shape == (24, 3)
    assert result["film_rgb_u8"].shape == (24, 3)
    assert result["digital_maximum_centre_channel_range"] <= 20
    assert result["film_maximum_centre_channel_range"] <= 20


def test_extractor_rejects_wrong_plot(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    wrong = tmp_path / "wrong.png"
    wrong.write_bytes(b"not a plot")
    with pytest.raises(ValueError, match="byte identity"):
        extract_exact_pair(wrong, wrong, config)


def test_frozen_observation_closes_nonbasic_matrix() -> None:
    config = deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    # Fixed vivid RGB patches plus neutral ramp exercise the actual RGB8-to-Lab path.
    digital = np.asarray(config["extraction"]["digital_patch_rgb_u8"], dtype=np.uint8)
    film = np.asarray(config["extraction"]["film_patch_rgb_u8"], dtype=np.uint8)
    result = evaluate_residual(digital, film, config)
    assert len(result["folds"]) == 6
    seen: list[int] = []
    for fold in result["folds"]:
        assert len(fold["development_indices"]) == 15
        assert len(fold["confirmation_indices"]) == 3
        assert not set(fold["development_indices"]) & set(
            fold["confirmation_indices"]
        )
        seen.extend(fold["confirmation_indices"])
    assert sorted(seen) == list(range(18))
    assert result["aggregate"]["identity"]["mean_confirmation_chroma_rmse"] == pytest.approx(
        11.417636176056632
    )
    assert result["aggregate"]["full_matrix"]["gain_over_global_chroma"] == pytest.approx(
        -0.06477981262856214
    )
    assert not result["automatic_pass"]
    assert result["selected_model"] == "none"
    assert result["decision"] == "close_controlled_chart_residual_without_capacity_rescue"


def test_evaluator_rejects_non_rgb8_shape() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="RGB8 24x3"):
        evaluate_residual(np.zeros((23, 3)), np.zeros((23, 3)), config)
