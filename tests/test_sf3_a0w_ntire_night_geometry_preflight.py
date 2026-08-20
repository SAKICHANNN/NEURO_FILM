from __future__ import annotations

import numpy as np

from src.real_film.ntire_night_geometry_preflight import (
    _rank_gray,
    match_facts,
    structural_view,
    validate_metadata,
)


def _config() -> dict[str, object]:
    return {
        "metadata_contract": {
            "required_exact_keys": [
                "as_shot_neutral",
                "black_level",
                "cfa_pattern",
                "huawei_bounds",
                "noise_profile",
                "orientation",
                "white_level",
            ],
            "as_shot_neutral_shape": [3],
            "black_level_shape": [4],
            "cfa_pattern_shape": [4],
            "huawei_bounds_shape": [4],
            "orientation_allowed": [1],
            "noise_profile_minimum_values": 2,
        }
    }


def _metadata() -> dict[str, object]:
    return {
        "as_shot_neutral": [0.5, 1.0, 0.75],
        "black_level": [64, 64, 64, 64],
        "cfa_pattern": [0, 1, 1, 2],
        "huawei_bounds": [0, 16, 0, 16],
        "noise_profile": [0.1, 0.01],
        "orientation": 1,
        "white_level": 1023,
    }


def test_metadata_contract_accepts_exact_observed_keys() -> None:
    validate_metadata(_metadata(), _config())


def test_structural_view_reconstructs_fixed_geometry() -> None:
    raw = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 900 + 64
    geometry = {
        "projective_pre_resize_factor": 1,
        "projective_matrix": np.eye(3).tolist(),
        "projective_output_width": 32,
        "projective_output_height": 32,
        "horizontal_flip": False,
        "pre_bounds_resize_width": 16,
        "pre_bounds_resize_height": 16,
        "upper_crop_start": 0,
        "final_width": 16,
        "final_height": 16,
    }
    view = structural_view(raw, _metadata(), geometry)
    assert view.shape == (16, 16, 3)
    assert np.isfinite(view).all()


def test_sift_identity_beats_unrelated_texture() -> None:
    rng = np.random.default_rng(20260821)
    base = (rng.random((256, 256)) * 255).astype(np.uint8)
    unrelated = (rng.random((256, 256)) * 255).astype(np.uint8)
    registration = {
        "sift_nfeatures": 2048,
        "ratio_test": 0.75,
        "ransac_reprojection_pixels": 3.0,
        "working_size": 256,
    }
    correct = match_facts(base, base.copy(), registration)
    wrong = match_facts(base, unrelated, registration)
    assert correct["inlier_ratio"] > wrong["inlier_ratio"]
    assert correct["maximum_corner_displacement_pixels"] < 1e-3


def test_rank_gray_is_deterministic() -> None:
    image = np.arange(32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
    assert np.array_equal(_rank_gray(image, 16), _rank_gray(image.copy(), 16))
