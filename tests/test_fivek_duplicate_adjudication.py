from __future__ import annotations

import csv
import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from src.eval.fivek_duplicate_adjudication import (
    compare_source_images,
    run_adjudication,
)


def _write_image(path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def _write_json(path, payload) -> str:
    data = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _protocol() -> dict:
    return {
        "maximum_dhash_hamming": 4,
        "maximum_phash_hamming": 8,
        "minimum_normalized_luma_correlation": 0.85,
        "orb": {
            "maximum_orb_side": 512,
            "maximum_features": 1000,
            "scale_factor": 1.2,
            "levels": 8,
            "ratio_test": 0.75,
            "minimum_mutual_ratio_matches": 12,
            "ransac_reprojection_pixels": 3.0,
            "ransac_maximum_iterations": 2000,
            "ransac_confidence": 0.995,
            "minimum_ransac_inliers": 12,
            "minimum_ransac_inlier_fraction": 0.35,
            "opencv_rng_seed": 20260801,
        },
    }


def test_multievidence_confirms_a_tone_changed_copy(tmp_path) -> None:
    yy, xx = np.mgrid[:192, :256]
    rgb = np.stack(
        [
            (xx * 7 + yy * 3) % 256,
            (xx * 2 + yy * 11) % 256,
            ((xx // 12 % 2) * 170 + (yy // 17 % 2) * 70) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    changed = np.clip(rgb.astype(np.float32) * 0.72 + 21.0, 0, 255).astype(
        np.uint8
    )
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_image(first, rgb)
    _write_image(second, changed)
    result = compare_source_images(first, second, _protocol())
    assert result["confirmed_duplicate"] is True
    assert result["phash_confirmed"] or result["orb"]["confirmed"]


def test_multievidence_orb_confirms_a_cropped_structured_copy(tmp_path) -> None:
    rng = np.random.default_rng(20260801)
    rgb = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    for index in range(24):
        center = (20 + (index * 47) % 280, 20 + (index * 31) % 200)
        cv2.circle(rgb, center, 4 + index % 11, (255, 255, 255), 2)
    cropped = cv2.resize(
        rgb[24:216, 32:288], (320, 240), interpolation=cv2.INTER_LINEAR
    )
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_image(first, rgb)
    _write_image(second, cropped)
    result = compare_source_images(first, second, _protocol())
    assert result["orb"]["confirmed"] is True
    assert result["confirmed_duplicate"] is True


def test_multievidence_rejects_low_information_dhash_collision(tmp_path) -> None:
    height, width = 192, 256
    first_luma = np.linspace(10, 245, height, dtype=np.uint8)[:, None]
    second_luma = (
        128
        + 100
        * np.sin(np.linspace(0, 6 * np.pi, height, dtype=np.float32))
    ).clip(0, 255).astype(np.uint8)[:, None]
    first_rgb = np.repeat(np.repeat(first_luma, width, axis=1)[..., None], 3, axis=2)
    second_rgb = np.repeat(
        np.repeat(second_luma, width, axis=1)[..., None], 3, axis=2
    )
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_image(first, first_rgb)
    _write_image(second, second_rgb)
    result = compare_source_images(first, second, _protocol())
    assert result["dhash_hamming"] <= 4
    assert not (
        result["phash_hamming"] <= 8
        and result["normalized_luma_correlation"] >= 0.85
    )
    assert result["orb"]["confirmed"] is False
    assert result["confirmed_duplicate"] is False


def test_runner_uses_only_declared_source_paths(tmp_path, monkeypatch) -> None:
    prior_image = tmp_path / "prior.png"
    development_image = tmp_path / "development.png"
    confirmation_image = tmp_path / "confirmation.png"
    for path in (prior_image, development_image, confirmation_image):
        path.write_bytes(b"source")
    parent_manifest = {
        "rows": [
            {
                "pair_id": "fresh-prior",
                "source_path": str(confirmation_image),
                "target_path": str(tmp_path / "forbidden-target-a.tif"),
            },
            {
                "pair_id": "development",
                "source_path": str(development_image),
                "target_path": str(tmp_path / "forbidden-target-b.tif"),
            },
            {
                "pair_id": "confirmation",
                "source_path": str(confirmation_image),
                "target_path": str(tmp_path / "forbidden-target-c.tif"),
            },
        ]
    }
    parent_report = {
        "automatic_pass": False,
        "perceptual_pairs": [
            {"existing_pair_id": "old", "fresh_pair_id": "fresh-prior"}
        ],
        "internal_perceptual_pairs": [
            {
                "development_pair_id": "development",
                "confirmation_pair_id": "confirmation",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    manifest_sha = _write_json(manifest_path, parent_manifest)
    report_sha = _write_json(report_path, parent_report)
    prior_csv = tmp_path / "prior.csv"
    with prior_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "path"])
        writer.writeheader()
        writer.writerow({"id": "old", "path": str(prior_image)})
    prior_sha = hashlib.sha256(prior_csv.read_bytes()).hexdigest()
    observed_paths = []

    def fake_compare(first, second, protocol):
        observed_paths.extend([first, second])
        assert "forbidden-target" not in str(first)
        assert "forbidden-target" not in str(second)
        return {
            "first_path": str(first),
            "first_sha256": "a" * 64,
            "second_path": str(second),
            "second_sha256": "b" * 64,
            "dhash_hamming": 4,
            "phash_hamming": 20,
            "normalized_luma_correlation": 0.0,
            "phash_confirmed": False,
            "orb": {"confirmed": False},
            "confirmed_duplicate": False,
        }

    monkeypatch.setattr(
        "src.eval.fivek_duplicate_adjudication.compare_source_images",
        fake_compare,
    )
    config = {
        "status": "contract_frozen_implementation_ready",
        "adaptive_successor_after_bq0s2_failure": True,
        "experiment_id": "test",
        "parent_report": {"path": report_path.name, "sha256": report_sha},
        "parent_manifest": {
            "path": manifest_path.name,
            "sha256": manifest_sha,
        },
        "prior_manifests": [
            {
                "path": prior_csv.name,
                "sha256": prior_sha,
                "format": "csv",
                "expected_rows": 1,
                "id_field": "id",
                "source_path_field": "path",
            }
        ],
        "expected_candidates": {"prior_pool": 1, "internal_split": 1},
        "protocol": {"maximum_dhash_hamming": 4},
        "target_pixels_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": "test",
    }
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    result = run_adjudication(
        root=tmp_path,
        config=config,
        config_path=config_path,
        output_path=tmp_path / "output" / "report.json",
        software_commit="c" * 40,
    )
    assert result["automatic_pass"] is True
    assert result["target_pixels_accessed"] is False
    assert len(observed_paths) == 4
