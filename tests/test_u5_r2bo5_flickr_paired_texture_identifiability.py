from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.eval.flickr_paired_texture_identifiability import (
    FlickrPairedTextureSupportError,
    analyze_pair,
    evaluate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bo5_flickr_paired_texture_identifiability_v1.json"


def test_pair_analysis_is_deterministic_and_finite() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))["analysis"]
    height = width = 192
    yy, xx = np.mgrid[:height, :width]
    base = 0.30 + 0.15 * xx / width + 0.1 * yy / height
    digital = np.stack((base, base * 0.98, base * 1.02), axis=2)
    rng = np.random.default_rng(7)
    shared = cv2.GaussianBlur(rng.normal(0.0, 0.012, (height, width)), (0, 0), 0.7)
    chroma = cv2.GaussianBlur(rng.normal(0.0, 0.004, (height, width)), (0, 0), 1.1)
    film = digital + np.stack((shared + chroma, shared, shared - chroma), axis=2)
    digital_u8 = np.rint(np.clip(digital, 0.0, 1.0) * 255).astype(np.uint8)
    film_u8 = np.rint(np.clip(film, 0.0, 1.0) * 255).astype(np.uint8)
    first = analyze_pair(digital_u8, film_u8, np.eye(3), config)
    second = analyze_pair(digital_u8, film_u8, np.eye(3), config)
    assert first == second
    assert first["patches"] >= config["minimum_patches_per_scene"]
    assert np.all(np.isfinite(list(first["residual_energy_log_luma_rg_bg"])))
    assert first["film_to_digital_luma_highpass_energy_ratio"] > 1.0


def test_formal_report_does_not_load_confirmation_pixels() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = evaluate(ROOT, config)
    assert report["confirmation_pixels_loaded"] is False
    assert report["metrics"]["sealed_confirmation_scenes_not_loaded"] == 12
    assert all(int(row["scene_id"]) % 4 != 0 for row in report["rows"])
    assert report["training_allowed"] is False
    assert report["operator_fitting_allowed"] is False


def test_small_pair_has_explicit_support_failure() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))["analysis"]
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    try:
        analyze_pair(image, image, np.eye(3), config)
    except FlickrPairedTextureSupportError as error:
        assert "patches" in str(error)
    else:
        raise AssertionError("small input must not fabricate patch support")
