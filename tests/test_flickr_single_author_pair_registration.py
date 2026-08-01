from __future__ import annotations

import cv2
import numpy as np

from src.eval.flickr_single_author_pair_registration import register_pair


def _contract() -> dict:
    return {
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


def _textured(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    for index in range(30):
        centre = (int(rng.integers(20, 620)), int(rng.integers(20, 460)))
        cv2.circle(image, centre, int(rng.integers(5, 25)), tuple(int(v) for v in rng.integers(0, 256, 3)), -1)
        cv2.putText(image, str(index), centre, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def test_register_pair_recovers_perspective_and_is_repeat_exact() -> None:
    digital = _textured()
    truth = np.asarray([[0.98, 0.015, 11.0], [-0.01, 1.01, 8.0], [0.00002, -0.00003, 1.0]])
    film = cv2.warpPerspective(digital, truth, (640, 480))
    first_h, first = register_pair(digital, film, _contract())
    second_h, second = register_pair(digital, film, _contract())
    assert first["registration_gate_passed"] is True
    assert second == first
    np.testing.assert_array_equal(second_h, first_h)
    np.testing.assert_allclose(first_h / first_h[2, 2], truth, atol=0.05)


def test_register_pair_rejects_unrelated_images() -> None:
    _, result = register_pair(_textured(1), _textured(2), _contract())
    assert result["registration_gate_passed"] is False
    assert result["failure_reason"] in {"insufficient_mutual_matches", "registration_quality_gate"}
