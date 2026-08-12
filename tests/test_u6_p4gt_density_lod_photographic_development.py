import json
from pathlib import Path

import numpy as np

from src.film_physics.density_lod_residual import apply_density_lod_residual

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gt_density_lod_photographic_development_v1.json"


def test_p4gt_freezes_density_lod_mechanism_before_execution():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["candidate"]["residual_projection"] == "density-lod-multiplicative"
    assert contract["candidate"]["pixel_aperture_kernel"] == [
        0.0625,
        0.25,
        0.375,
        0.25,
        0.0625,
    ]
    assert contract["source"]["freshness"].startswith("consumed P4GR/P4GS")
    assert contract["execution"]["same_cohort_rescue_allowed"] is False


def test_density_lod_residual_preserves_ratios_and_suppresses_endpoints():
    base = np.empty((7, 9, 3), dtype=np.float32)
    base[:2] = (0.0, 0.0, 0.0)
    base[2:5] = (0.2, 0.4, 0.8)
    base[5:] = (1.0, 1.0, 1.0)
    reference = np.full_like(base, 0.5)
    physical = reference.copy()
    physical[3, 4] = 0.2
    output, diagnostics = apply_density_lod_residual(base, physical, reference)
    assert np.array_equal(output[:2], base[:2])
    assert np.array_equal(output[5:], base[5:])
    assert np.all(output >= 0.0) and np.all(output <= 1.0)
    assert np.allclose(output[2:5, :, 0] / output[2:5, :, 1], 0.5)
    assert np.allclose(output[2:5, :, 1] / output[2:5, :, 2], 0.5)
    assert diagnostics["integrated_density_rms"] < diagnostics["unbounded_common_density_rms"]
    assert diagnostics["hard_clipping_used"] == 0.0


def test_density_lod_residual_is_partition_exact_with_two_row_halo():
    rng = np.random.default_rng(184)
    base = rng.uniform(0.05, 0.95, size=(17, 13, 3)).astype(np.float32)
    reference = rng.uniform(0.2, 0.8, size=base.shape).astype(np.float32)
    physical = np.clip(reference * rng.uniform(0.8, 1.2, size=base.shape), 1e-4, 1.0)
    full, _ = apply_density_lod_residual(base, physical, reference)
    pieces = []
    for start, stop in ((0, 5), (5, 12), (12, 17)):
        halo_start = max(0, start - 2)
        halo_stop = min(17, stop + 2)
        window, _ = apply_density_lod_residual(
            base[halo_start:halo_stop],
            physical[halo_start:halo_stop],
            reference[halo_start:halo_stop],
        )
        pieces.append(window[start - halo_start : stop - halo_start])
    assert np.array_equal(full, np.concatenate(pieces))
