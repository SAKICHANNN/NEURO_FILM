from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.color_precision_page_identifiability import (
    ColorPrecisionPageIdentifiabilityError,
    aggregate_page_bags,
    bind_extracted_images,
    evaluate_page_identifiability,
    image_descriptors,
    parse_pdfimages_objects,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (
        ROOT
        / "configs/u5_r2bd1_color_precision_page_identifiability_v1.json"
    ).read_text(encoding="utf-8")
)


def test_contract_forbids_exposure_fit_training_and_modes() -> None:
    assert CONFIG["sample_unit"]["exposure_assignment_allowed"] is False
    assert CONFIG["extractor"]["embedded_profile_semantics"].startswith(
        "not_preserved"
    )
    forbidden = set(CONFIG["forbidden_actions"])
    assert {
        "assigning_exposure_ev_to_images",
        "operator_fitting",
        "training",
        "latent_mode_clustering",
    } <= forbidden


def test_object_parser_and_binding_collapse_duplicate_pdf_object(
    tmp_path: Path,
) -> None:
    listing = """
page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio
1 0 image 3112 2082 icc 3 8 image yes 9 0 778 778 15M 80%
1 1 image 3112 2082 icc 3 8 image yes 9 0 778 778 15M 80%
1 2 image 2100 2100 icc 3 8 image yes 10 0 1419 1419 880K 7%
"""
    rows = parse_pdfimages_objects(
        listing,
        scanner_id="frontier",
        minimum_width=2500,
        minimum_height=1800,
    )
    prefix = tmp_path / "frontier"
    payload = np.full((8, 8, 3), 127, dtype=np.uint8)
    Image.fromarray(payload).save(tmp_path / "frontier-000.png")
    Image.fromarray(payload).save(tmp_path / "frontier-001.png")
    bound = bind_extracted_images(rows, prefix=prefix)
    assert len(bound) == 1
    assert bound[0]["object_id"] == 9


def test_object_binding_rejects_one_object_with_different_pixels(
    tmp_path: Path,
) -> None:
    listing = """
1 0 image 3112 2082 icc 3 8 image yes 9 0 778 778 15M 80%
1 1 image 3112 2082 icc 3 8 image yes 9 0 778 778 15M 80%
"""
    rows = parse_pdfimages_objects(
        listing,
        scanner_id="frontier",
        minimum_width=2500,
        minimum_height=1800,
    )
    prefix = tmp_path / "frontier"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(tmp_path / "frontier-000.png")
    Image.new("RGB", (8, 8), (1, 0, 0)).save(tmp_path / "frontier-001.png")
    with pytest.raises(
        ColorPrecisionPageIdentifiabilityError, match="different pixels"
    ):
        bind_extracted_images(rows, prefix=prefix)


def test_image_descriptors_are_finite_and_exposure_normalization_is_stable(
    tmp_path: Path,
) -> None:
    gradient = np.linspace(0.05, 0.8, 64, dtype=np.float64).reshape(8, 8)
    image = np.stack((gradient, gradient * 0.8, gradient * 0.6), axis=-1)
    bright = np.clip(image * 1.18, 0.0, 1.0)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(first)
    Image.fromarray(np.rint(bright * 255).astype(np.uint8)).save(second)
    raw_a, normalized_a = image_descriptors(first, resize=(256, 256))
    raw_b, normalized_b = image_descriptors(second, resize=(256, 256))
    assert raw_a.shape == normalized_a.shape == (59,)
    assert np.isfinite(normalized_a).all()
    assert np.linalg.norm(raw_a - raw_b) > 0.5
    assert np.linalg.norm(normalized_a[6:11] - normalized_b[6:11]) < 0.02


def _paired_rows(*, normalized_permutation: bool = False) -> list[dict]:
    rng = np.random.default_rng(5)
    rows = []
    dimension = 177
    permutation = np.roll(np.arange(12), 1)
    bases = rng.normal(size=(12, dimension))
    for scanner_index, scanner in enumerate(("frontier", "noritsu")):
        for index in range(12):
            normalized_index = (
                int(permutation[index])
                if scanner == "noritsu" and normalized_permutation
                else index
            )
            rows.append(
                {
                    "scanner_id": scanner,
                    "variant_id": f"stock_{index}/normal",
                    "raw_descriptor": (
                        bases[index] + scanner_index * 0.001
                    ).tolist(),
                    "basic_normalized_descriptor": (
                        bases[normalized_index] + scanner_index * 0.001
                    ).tolist(),
                }
            )
    return rows


def test_evaluator_passes_paired_signal_and_keeps_claims_closed() -> None:
    config = deepcopy(CONFIG)
    config["evaluation"]["permutation"]["draws"] = 999
    report = evaluate_page_identifiability(_paired_rows(), config)
    assert report["gates"]["all_passed"] is True
    assert report["branch"] == "raw_and_normalized_pass"
    assert report["exposure_assignment_performed"] is False
    assert report["operator_fitting_performed"] is False
    assert report["training_performed"] is False
    assert report["latent_mode_clustering_performed"] is False


def test_evaluator_detects_raw_only_shortcut() -> None:
    config = deepcopy(CONFIG)
    config["evaluation"]["permutation"]["draws"] = 999
    report = evaluate_page_identifiability(
        _paired_rows(normalized_permutation=True), config
    )
    assert report["gates"]["raw_passed"] is True
    assert report["gates"]["basic_normalized_passed"] is False
    assert report["branch"] == "raw_pass_normalized_fail"


def test_page_bag_requires_exact_label_identity(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), (30, 60, 90)).save(image)
    rows = [
        {
            "scanner_id": "frontier",
            "page_index": 1,
            "image_number": 0,
            "object_id": 1,
            "object_generation": 0,
            "path": image.as_posix(),
            "bytes": image.stat().st_size,
            "sha256": "0" * 64,
        }
    ]
    with pytest.raises(
        ColorPrecisionPageIdentifiabilityError, match="page identities"
    ):
        aggregate_page_bags(rows, [], resize=(256, 256))
