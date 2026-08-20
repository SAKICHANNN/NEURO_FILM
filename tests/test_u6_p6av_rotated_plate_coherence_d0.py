from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import tifffile

from src.eval.rotated_plate_coherence_d0 import (
    _load_u16,
    _midrank,
    _register_orientation,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p6av_contract_keeps_method_and_claim_narrow() -> None:
    contract = load_contract(
        ROOT / "configs/u6_p6av_rotated_plate_coherence_d0_v1.json"
    )
    assert contract["analysis"]["patch_grid"] == [4, 4]
    assert contract["gates"]["required_patches"] == 16
    assert "P6L" in contract["forbidden"][0]
    assert "isolated emulsion NPS" in contract["claim_ceiling"]


def test_p6aw_keeps_p6av_protocol_values_exact() -> None:
    first = load_contract(ROOT / "configs/u6_p6av_rotated_plate_coherence_d0_v1.json")
    second = load_contract(
        ROOT / "configs/u6_p6aw_barnard_rotated_plate_coherence_d0_v1.json"
    )
    assert second["registration"] == first["registration"]
    first_analysis = {k: v for k, v in first["analysis"].items() if k != "luminance"}
    second_analysis = {k: v for k, v in second["analysis"].items() if k != "luminance"}
    assert second_analysis == first_analysis
    assert second["gates"] == first["gates"]
    assert second["source"]["reference"]["required_pages"] == 2
    assert second["source"]["reference"]["required_reduced_page_indices"] == [1]


def test_p6aw_selects_full_resolution_page_from_reduced_thumbnail_tiff(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.tif"
    full = np.arange(48, dtype=np.uint16).reshape(6, 8)
    reduced = full[::2, ::2]
    with tifffile.TiffWriter(path) as writer:
        writer.write(full)
        writer.write(reduced, subfiletype=1)
    payload = path.read_bytes()
    values, facts = _load_u16(
        path,
        {
            "path": path.name,
            "bytes": len(payload),
            "md5": hashlib.md5(payload).hexdigest(),
            "required_pages": 2,
            "primary_page_index": 0,
            "required_reduced_page_indices": [1],
        },
    )
    assert np.array_equal(values, full)
    assert facts["pages"] == 2
    assert facts["primary_page_index"] == 0


def test_p6aw_converts_rgb16_primary_page_with_frozen_equal_weight_rule(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source_rgb.tif"
    full = np.array([[[0, 1, 2], [10, 11, 12]]], dtype=np.uint16)
    with tifffile.TiffWriter(path) as writer:
        writer.write(full)
        writer.write(full[:, ::2], subfiletype=1)
    payload = path.read_bytes()
    values, facts = _load_u16(
        path,
        {
            "path": path.name,
            "bytes": len(payload),
            "md5": hashlib.md5(payload).hexdigest(),
            "required_pages": 2,
            "primary_page_index": 0,
            "required_reduced_page_indices": [1],
            "primary_samples_per_pixel": 3,
            "luminance_conversion": "equal-rgb-rounded-nearest-u16",
        },
    )
    assert np.array_equal(values, np.array([[1, 11]], dtype=np.uint16))
    assert facts["primary_samples_per_pixel"] == 3


def test_midrank_is_tie_stable() -> None:
    values = np.array([[0, 0], [10, 20]], dtype=np.uint16)
    result = _midrank(values)
    assert np.array_equal(result, np.array([[0.25, 0.25], [0.625, 0.875]]))


def test_registration_selects_supported_quarter_turn() -> None:
    rng = np.random.default_rng(41)
    reference = np.zeros((512, 512), dtype=np.uint16)
    for _ in range(120):
        y, x = rng.integers(20, 492, size=2)
        cv = int(rng.integers(8000, 62000))
        reference[y - 3 : y + 4, x - 3 : x + 4] = cv
    rotated = np.rot90(reference, -1)
    config = {
        "orientation_candidates_degrees": [90, -90],
        "maximum_features": 8000,
        "maximum_side_pixels": 512,
        "ratio_threshold": 0.7,
        "ransac_reprojection_threshold_pixels": 3.0,
    }
    homography, diagnostics = _register_orientation(reference, rotated, config)
    assert homography is not None
    assert diagnostics["orientation_degrees"] == 90
    assert diagnostics["inliers"] > 50
    assert diagnostics not in diagnostics["candidates"]
