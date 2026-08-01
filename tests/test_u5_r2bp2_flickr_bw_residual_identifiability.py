from __future__ import annotations

import numpy as np
import cv2

from src.eval.flickr_bw_residual_identifiability import analyze_pair


CONTRACT = {
    "valid_mask_erosion_pixels": 4,
    "encoded_boundary_code_minimum": 3,
    "encoded_boundary_code_maximum": 252,
    "tone_quantiles": 17,
    "prefilter_sigma_pixels": 0.8,
    "highpass_sigma_pixels": 2.0,
    "patch_size_pixels": 48,
    "patch_stride_pixels": 48,
    "maximum_patches_per_scene": 16,
    "minimum_patches_per_scene": 6,
    "maximum_patch_gradient_quantile": 0.5,
    "log_floor": 1.0 / 255.0,
    "shift_control_pixels": 4,
    "radial_frequency_edges_nyquist": [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.000001],
}


def test_analyze_pair_detects_added_high_frequency_energy() -> None:
    rng = np.random.default_rng(29)
    source = cv2.GaussianBlur(
        rng.uniform(0.2, 0.8, size=(480, 720)), (0, 0), 3.0
    )
    film = np.clip(0.08 + 0.84 * source + rng.normal(0.0, 0.04, source.shape), 0.02, 0.98)
    result = analyze_pair(source, film, np.eye(3), CONTRACT)
    assert result["patches"] >= 6
    assert result["film_to_basic_highpass_energy_ratio"] > 1.0
    assert len(result["film_minus_basic_radial_psd"]) == 6
