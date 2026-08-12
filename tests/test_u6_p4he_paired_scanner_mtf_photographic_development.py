from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.paired_scanner_mtf_photographic_development import load_contract
from src.film_physics.spatial_response import (
    SpatialResponseProfile,
    apply_scanner_mtf,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4he_paired_scanner_mtf_photographic_development_v1.json"


def test_contract_pins_p4hd_and_exact_p4fa_scanner() -> None:
    payload = load_contract(CONTRACT)
    candidate = payload["candidate"]
    p4fa = json.loads(
        (ROOT / payload["parents"]["p4fa_contract"]["path"]).read_text("utf-8")
    )
    assert candidate["scanner_mtf_sigma_pixels_rgb"] == p4fa["spatial_profiles"][
        "scanner_mtf_sigma_pixels_rgb"
    ]
    assert candidate["gaussian_truncate"] == p4fa["spatial_profiles"][
        "gaussian_truncate"
    ]
    assert candidate["baseline_and_candidate_share_downstream_stage"] is True
    assert payload["execution"]["post_result_retuning_allowed"] is False


def test_paired_scanner_stage_is_common_and_nonidentity() -> None:
    profile = SpatialResponseProfile(
        pixel_pitch_um=1.0,
        forward_scatter_sigma_um_rgb=(0.0, 0.0, 0.0),
        development_adjacency_sigma_um_rgb=(0.0, 0.0, 0.0),
        development_adjacency_gain_rgb=(0.0, 0.0, 0.0),
        dye_diffusion_sigma_um_rgb=(0.0, 0.0, 0.0),
        scanner_mtf_sigma_um_rgb=(0.7, 0.7, 0.7),
        gaussian_truncate=3.0,
    )
    source = np.zeros((17, 19, 3), dtype=np.float64)
    source[8, 9] = (1.0, 0.5, 0.25)
    candidate = source.copy()
    candidate[7, 9] += (0.02, 0.01, 0.005)
    baseline_output = apply_scanner_mtf(source, profile)
    candidate_output = apply_scanner_mtf(candidate, profile)
    assert not np.array_equal(baseline_output, source)
    assert np.max(np.abs(candidate_output - baseline_output)) > 0.0
    assert np.all((baseline_output >= 0.0) & (baseline_output <= 1.0))
    assert np.all((candidate_output >= 0.0) & (candidate_output <= 1.0))
