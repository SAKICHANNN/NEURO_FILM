from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.spektrafilm_spatial_dir import (
    matched_local_contrast,
    new_isolated_red_speckle_fraction,
    paired_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/u5_r2ad1_spektrafilm_spatial_dir_ablation_v1.json").read_text(
        encoding="utf-8"
    )
)


def _image() -> np.ndarray:
    y, x = np.mgrid[:48, :64]
    return np.stack(
        (
            0.15 + 0.7 * x / 63,
            0.1 + 0.8 * y / 47,
            0.2 + 0.5 * (x + y) / 110,
        ),
        axis=-1,
    )


def test_local_contrast_match_recovers_frozen_family_member() -> None:
    off = _image()
    from scipy.ndimage import gaussian_filter

    detail = off - gaussian_filter(off, sigma=(2.0, 2.0, 0.0), mode="reflect")
    on = off + 0.4 * detail
    matched, fit = matched_local_contrast(
        off, on, CONFIG["matched_local_contrast_control"]
    )
    assert fit["sigma_pixels"] == 2.0
    assert abs(fit["alpha"] - 0.4) < 1e-10
    assert fit["explained_rgb_energy_fraction"] > 1.0 - 1e-12
    np.testing.assert_allclose(matched, on, rtol=0.0, atol=1e-12)


def test_red_speckle_detector_is_paired_and_isolated() -> None:
    off = np.full((17, 17, 3), 0.4, dtype=np.float64)
    on = off.copy()
    on[8, 8] = (0.9, 0.2, 0.2)
    assert new_isolated_red_speckle_fraction(off, on) == 1.0 / (17 * 17)
    assert new_isolated_red_speckle_fraction(on, on) == 0.0


def test_paired_metrics_are_finite_for_identical_neutral_pair() -> None:
    image = _image()
    metrics = paired_metrics(image, image, image, CONFIG)
    assert metrics["finite"]
    assert metrics["bounded_0_1"]
    assert metrics["spatial_effect_delta_e76"] == 0.0
    assert metrics["new_hard_clipping_fraction_vs_spatial_off"] == 0.0
    assert metrics["new_isolated_red_speckle_fraction"] == 0.0
