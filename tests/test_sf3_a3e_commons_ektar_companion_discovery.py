from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.real_film.commons_ektar_companion import (
    flickr_identity,
    metadata_candidate,
    pixel_gate,
    registered_similarity,
    title_jaccard,
)


def _source() -> dict:
    return {
        "page_id": 10,
        "title": "File:Old station at Pixley California.jpg",
        "author_raw_html": '<a href="https://www.flickr.com/people/abc@N01">A</a>',
        "credit_raw_html": '<a href="https://www.flickr.com/photos/abc@N01/123">x</a>',
        "date_time_original": "2020-02-12 18:17",
    }


def _candidate(**updates) -> dict:
    row = {
        "page_id": 11,
        "title": "File:Pixley California old station digital.jpg",
        "license_short_name": "CC BY 4.0",
        "mime": "image/jpeg",
        "width": 1200,
        "height": 800,
        "camera_model": "Nikon D850",
        "date_time_original": "2020-02-12 18:20:00",
        "description_raw_html": "Digital camera study",
    }
    row.update(updates)
    return row


def _policy() -> dict:
    return {
        "allowed_licenses": ["CC BY 4.0"],
        "allowed_mime": ["image/jpeg"],
        "maximum_capture_time_delta_seconds": 900,
        "minimum_title_token_jaccard": 0.15,
        "minimum_short_dimension": 512,
        "digital_camera_model_required": True,
        "forbidden_digital_model_tokens": ["scanner", "noritsu"],
        "forbidden_candidate_text_tokens": ["ektar", "film scan"],
    }


def test_flickr_identity_requires_one_exact_identity() -> None:
    assert flickr_identity(_source()) == "abc@n01"
    assert flickr_identity({"author_raw_html": "none", "credit_raw_html": ""}) == ""


def test_metadata_candidate_accepts_independent_digital_companion() -> None:
    row = metadata_candidate(_source(), _candidate(), _policy())
    assert row is not None
    assert row["capture_time_delta_seconds"] == 180.0
    assert row["title_token_jaccard"] == title_jaccard(_source()["title"], _candidate()["title"])


def test_metadata_candidate_rejects_scan_film_or_unrelated_time() -> None:
    assert metadata_candidate(_source(), _candidate(camera_model="Noritsu scanner"), _policy()) is None
    assert metadata_candidate(_source(), _candidate(description_raw_html="Ektar film scan"), _policy()) is None
    assert metadata_candidate(_source(), _candidate(date_time_original="2020-02-13 18:20:00"), _policy()) is None


def test_registered_similarity_recovers_projective_same_scene(tmp_path: Path) -> None:
    base = np.zeros((480, 640), dtype=np.uint8)
    rng = np.random.default_rng(17)
    for _ in range(120):
        x, y = rng.integers(20, 620), rng.integers(20, 460)
        cv2.circle(base, (int(x), int(y)), int(rng.integers(3, 12)), int(rng.integers(80, 255)), -1)
    cv2.putText(base, "SAME SCENE", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 3)
    matrix = np.float32([[1, 0, 9], [0, 1, -7]])
    moved = cv2.warpAffine(base, matrix, (640, 480))
    source = tmp_path / "source.png"
    candidate = tmp_path / "candidate.png"
    Image.fromarray(base).save(source)
    Image.fromarray(moved).save(candidate)
    metrics = registered_similarity(source, candidate)
    assert metrics["sift_matches"] >= 24
    assert metrics["homography_inliers"] >= 16
    assert metrics["registered_gradient_ncc"] >= 0.8


def test_pixel_gate_rejects_near_duplicate_even_when_registration_passes() -> None:
    gates = {
        "minimum_sift_matches": 24,
        "minimum_homography_inliers": 16,
        "minimum_homography_inlier_ratio": 0.35,
        "maximum_median_reprojection_error_pixels": 3.0,
        "minimum_registered_gradient_ncc": 0.45,
        "near_duplicate_hamming_threshold": 4,
    }
    metrics = {
        "sift_matches": 30,
        "homography_inliers": 20,
        "homography_inlier_ratio": 0.6,
        "median_reprojection_error_pixels": 0.5,
        "registered_gradient_ncc": 0.9,
        "dhash_hamming": 2,
    }
    checks = pixel_gate(metrics, gates)
    assert not checks["not_near_duplicate"]
    assert not all(checks.values())
