from __future__ import annotations

import numpy as np

from src.eval.flickr_single_author_pair_registration import register_pair


CONTRACT = {
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": [8, 8],
    "maximum_features": 6000,
    "ratio_threshold": 0.75,
    "ransac_reprojection_threshold_pixels": 4.0,
    "minimum_mutual_good_matches": 20,
    "minimum_inliers": 12,
    "minimum_inlier_fraction": 0.35,
    "maximum_median_inlier_reprojection_error_pixels": 2.0,
    "maximum_p95_inlier_reprojection_error_pixels": 5.0,
    "minimum_target_overlap_fraction": 0.3,
}


def test_registration_recovers_simple_shift() -> None:
    rng = np.random.default_rng(17)
    base = rng.integers(0, 256, size=(480, 720, 3), dtype=np.uint8)
    film = np.roll(base, shift=(5, -7), axis=(0, 1))
    _, diagnostics = register_pair(base, film, CONTRACT)
    assert diagnostics["registration_gate_passed"] is True
    assert diagnostics["median_inlier_reprojection_error_pixels"] < 0.1
