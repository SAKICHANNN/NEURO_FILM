from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2aq3v_colorreference_recorder_patch_visual import (
    CONFIG_SHA256,
    load_config,
    sample_position,
    xyz_d50_to_srgb_preview,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq3v_colorreference_recorder_patch_visual_v1.json"
)


def test_contract_is_synthetic_and_not_product_rendering() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    assert config["ordinary_photo_application_allowed"] is False
    assert config["production_integration_allowed"] is False
    assert config["visual_gate"]["style_or_photo_claim_allowed"] is False


def test_sample_layout_matches_it8_rows() -> None:
    assert sample_position("A1") == (0, 0)
    assert sample_position("L22") == (11, 21)
    assert sample_position("GS24") == (12, 23)
    with pytest.raises(ValueError, match="column"):
        sample_position("A23")
    with pytest.raises(ValueError, match="unsupported"):
        sample_position("M1")


def test_xyz_preview_reports_clipping_before_clamp() -> None:
    encoded, report = xyz_d50_to_srgb_preview(
        np.asarray([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    )
    assert encoded.shape == (2, 3)
    assert np.min(encoded) >= 0.0
    assert np.max(encoded) <= 1.0
    assert report["preclamp_above_one_component_fraction"] > 0.0


def test_config_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["ordinary_photo_application_allowed"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(path, CONFIG_SHA256)
