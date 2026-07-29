from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from src.eval.real_uniform_grain_analysis import build_scan_signatures
from src.eval.real_uniform_grain_analysis import (
    UniformGrainAnalysisError,
    _validate_acquisition_manifest,
)


def test_build_scan_signatures_keeps_rgb_and_ir_distinct() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(size=(256, 256))
    rgb = np.stack(
        [
            20000 + 500 * gaussian_filter(base, sigma=sigma)
            for sigma in (0.8, 1.2, 2.0)
        ],
        axis=-1,
    ).astype(np.uint16)
    infrared = (
        30000 + 500 * gaussian_filter(base, sigma=3.0)
    ).astype(np.uint16)
    result = build_scan_signatures(
        rgb=rgb,
        infrared=infrared,
        pixel_contract={
            "crop_size_pixels": 96,
            "fixed_fractional_centers_yx": [
                [0.3, 0.3],
                [0.3, 0.7],
                [0.7, 0.3],
                [0.7, 0.7],
            ],
            "radial_nps_band_edges_cycles_per_pixel": (
                np.geomspace(1.0 / 32.0, 0.45, 9).tolist()
            ),
            "acf_lags_pixels_yx": [
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 4],
            ],
        },
    )
    assert result["combined_rgb_nps"].shape == (4, 24)
    assert result["channel_nps"]["infrared"].shape == (4, 8)
    assert result["channel_acf"]["red"].shape == (4, 4)
    assert not np.array_equal(
        result["channel_nps"]["red"],
        result["channel_nps"]["infrared"],
    )


def test_acquisition_manifest_must_bind_every_source_row() -> None:
    source = {
        "experiment_id": "source-v1",
        "selection": {"expected_total_bytes": 12},
        "files": [
            {
                "title": "File:one.tif",
                "film_stock_id": "stock-a",
                "path": "data/one.tif",
                "api_sha1": "1" * 40,
                "expected_bytes": 12,
                "allowed_use": "research",
            }
        ],
    }
    manifest = {
        "schema": (
            "neuro_film.u6_p4r_uniform_grain_acquisition_manifest.v1"
        ),
        "source_contract_id": "source-v1",
        "total_bytes": 12,
        "rows": [
            {
                "title": "File:one.tif",
                "film_stock_id": "stock-a",
                "path": "data/one.tif",
                "sha1": "1" * 40,
                "sha256": "2" * 64,
                "bytes": 12,
                "allowed_use": "research",
            }
        ],
    }
    _validate_acquisition_manifest(manifest, source)
    manifest["rows"][0]["sha1"] = "3" * 40
    try:
        _validate_acquisition_manifest(manifest, source)
    except UniformGrainAnalysisError as error:
        assert "row identity" in str(error)
    else:
        raise AssertionError("mutated acquisition identity must fail closed")
