from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_scanner_profile import (
    evaluate_scanner_profiles,
    load_contract,
)
from src.film_physics import ScannerProfile, apply_scanner_profile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6a_scanner_profile_boundary_v1.json"


def _identity() -> ScannerProfile:
    return ScannerProfile(
        profile_id="identity",
        illuminant_rgb=(1.0, 1.0, 1.0),
        spectral_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        local_flare_fraction=0.0,
        global_flare_fraction=0.0,
        flare_sigma_um=0.0,
        dmax_density_rgb=None,
        mtf_sigma_um_rgb=(0.0, 0.0, 0.0),
        shot_noise_variance_scale=0.0,
        read_noise_variance=0.0,
        seed=0,
    )


def test_identity_scanner_is_exact() -> None:
    values = np.linspace(0.001, 1.0, 300, dtype=np.float64).reshape(10, 10, 3)
    output = apply_scanner_profile(values, _identity(), pixel_pitch_um=1.0)
    assert np.array_equal(output, values)


def test_scanner_rejects_stage_reordering() -> None:
    values = np.ones((4, 4, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        apply_scanner_profile(
            values,
            _identity(),
            pixel_pitch_um=1.0,
            stages=("noise", "spectral"),
        )


def test_scanner_noise_is_repeat_exact_and_coordinate_stable() -> None:
    profile = ScannerProfile(
        profile_id="noise",
        illuminant_rgb=(1.0, 1.0, 1.0),
        spectral_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        local_flare_fraction=0.0,
        global_flare_fraction=0.0,
        flare_sigma_um=0.0,
        dmax_density_rgb=None,
        mtf_sigma_um_rgb=(0.0, 0.0, 0.0),
        shot_noise_variance_scale=1e-6,
        read_noise_variance=1e-7,
        seed=5,
    )
    values = np.full((16, 12, 3), 0.5, dtype=np.float64)
    full = apply_scanner_profile(values, profile, pixel_pitch_um=1.0)
    repeat = apply_scanner_profile(values, profile, pixel_pitch_um=1.0)
    region = apply_scanner_profile(
        values[4:12],
        profile,
        pixel_pitch_um=1.0,
        full_shape=values.shape[:2],
        origin_yx=(4, 0),
    )
    assert np.array_equal(full, repeat)
    assert np.array_equal(full[4:12], region)


def test_frozen_scanner_profiles_are_identifiable() -> None:
    report = evaluate_scanner_profiles(load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["stable_evidence_id"] == (
        "17fafa26d4c165c4f3a434d089ab1dae25b49d47fbbc98d3f94a90b2a7605b15"
    )
    assert all(report["decisions"].values())
