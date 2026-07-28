from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_neutral_gauged_invariance import (
    render_challenger,
    render_challenger_row_tiled,
    validate_contract,
)
from src.eval.physical_virtual_scan_sampling import (
    compile_virtual_scan_profile,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p7g_neutral_gauged_invariance_v1.json"
)


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p7g_contract_binds_passing_p7f_without_promotion() -> None:
    runtime, _ = validate_contract(ROOT, _config())
    assert set(_config()["tile_audit"]["real_ids"]) <= set(
        runtime.eligible_ids
    )


def test_virtual_scan_compiler_preserves_normalized_physical_scale() -> None:
    runtime, _ = validate_contract(ROOT, _config())
    high = compile_virtual_scan_profile(
        runtime.profile, sampling_dpi=4000
    )
    low = compile_virtual_scan_profile(
        runtime.profile, sampling_dpi=2000
    )
    for field in (
        "forward_scatter_sigma_um_rgb",
        "development_adjacency_sigma_um_rgb",
        "dye_diffusion_sigma_um_rgb",
        "scanner_mtf_sigma_um_rgb",
    ):
        high_pixels = np.asarray(getattr(high, field)) / high.pixel_pitch_um
        low_pixels = np.asarray(getattr(low, field)) / low.pixel_pitch_um
        assert np.array_equal(low_pixels * 2.0, high_pixels)


def test_row_tiled_challenger_matches_full_for_odd_partitions() -> None:
    runtime, gauge = validate_contract(ROOT, _config())
    source = np.random.default_rng(7).random((47, 61, 3))
    reference = render_challenger(
        source, runtime, gauge, sampling_dpi=4000
    )
    for tile_rows in (1, 8, 19, 99):
        for order in ("forward", "reverse"):
            tiled, _ = render_challenger_row_tiled(
                source,
                runtime,
                gauge,
                sampling_dpi=4000,
                tile_rows=tile_rows,
                order=order,
            )
            assert np.array_equal(tiled, reference)
